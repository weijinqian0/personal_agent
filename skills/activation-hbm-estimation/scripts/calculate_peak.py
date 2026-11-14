"""Sum unique per-rank storages by simultaneous phase; keep unknowns unknown."""

import argparse
import json
import sys

EXCLUDED = {"weights", "kv_cache"}
CATEGORIES = EXCLUDED | {"activation", "workspace", "communication", "graph_pool",
                         "persistent_buffer", "runtime", "allocator_extra"}


def calculate(data):
    if not isinstance(data.get("rank"), (str, int)) or isinstance(data["rank"], bool):
        raise ValueError("rank must identify the single rank represented by this ledger")
    phases = data.get("phases")
    if (not isinstance(phases, list) or not phases
            or any(not isinstance(p, str) or not p or p == "*" for p in phases)
            or len(set(phases)) != len(phases)):
        raise ValueError("phases must be distinct nonempty strings; '*' is reserved")
    coverage = data.get("coverage", {})
    for key in ("unassessed_components", "unassessed_phases"):
        if key not in coverage or not isinstance(coverage[key], list):
            raise ValueError(f"coverage.{key} must be an explicit list (possibly empty)")
        if any(not isinstance(s, str) or not s for s in coverage[key]):
            raise ValueError(f"coverage.{key} must contain nonempty strings")
    storages = data.get("storages")
    if not isinstance(storages, list):
        raise ValueError("storages must be a list")
    totals = {p: 0 for p in phases}
    unknowns = {p: [] for p in phases}
    seen, assumed, excluded = set(), [], []
    for s in storages:
        ident = s.get("id")
        if not isinstance(ident, str) or not ident or ident in seen:
            raise ValueError("storage IDs must be nonempty and unique; consolidate aliases first")
        seen.add(ident)
        category = s.get("category")
        if category not in CATEGORIES:
            raise ValueError(f"invalid category for {ident}")
        amount = s.get("bytes")
        if amount is not None and (type(amount) is not int or amount < 0):
            raise ValueError(f"{ident}: bytes must be a nonnegative integer or null")
        status = s.get("status")
        if status not in {"confirmed", "assumed", "unknown"}:
            raise ValueError(f"{ident}: status must be confirmed, assumed or unknown")
        if status == "unknown" and amount is not None:
            raise ValueError(f"{ident}: use assumed for conditional numeric amounts")
        if amount is None and status != "unknown":
            raise ValueError(f"{ident}: null bytes require unknown status")
        live = s.get("live_phases")
        if (not isinstance(live, list) or not live
                or any(not isinstance(p, str) for p in live)
                or len(set(live)) != len(live)
                or ("*" in live and live != ["*"])
                or any(p not in totals and p != "*" for p in live)):
            raise ValueError(f"{ident}: invalid live_phases")
        if category in EXCLUDED:
            excluded.append(ident)
            continue
        if status == "assumed":
            assumed.append(ident)
        for p in phases if live == ["*"] else live:
            if amount is None:
                unknowns[p].append(ident)
            else:
                totals[p] += amount
    complete = (not any(unknowns.values()) and not coverage["unassessed_components"]
                and not coverage["unassessed_phases"])
    peak = max(totals.values())
    return {
        "rank": data["rank"],
        "scope": "non-weight non-KV unique-storage ledger, not allocator measurement",
        "phase_known_subtotals": {
            p: {"bytes": n, "MiB": n/2**20, "GiB": n/2**30,
                "unknown_storages": unknowns[p]} for p, n in totals.items()
        },
        "known_subtotal_peak_bytes": peak,
        "known_subtotal_peak_phases": [p for p, n in totals.items() if n == peak],
        "ledger_complete_as_declared": complete,
        "complete_ledger_peak_bytes": peak if complete else None,
        "conditional_numeric_storages": assumed,
        "excluded_storage_ids": excluded,
        "coverage": coverage,
        "warning": "No automatic source, alias, lifecycle, pool-overlap or coverage verification. Completeness is the caller's declaration; conditional inputs are not measurements.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    try:
        with open(args.input, encoding="utf-8") as handle:
            result = calculate(json.load(handle))
    except (ValueError, TypeError, KeyError, AttributeError, OSError) as exc:
        print(f"Invalid ledger: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
