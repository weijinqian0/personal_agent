#!/usr/bin/env python3
"""Audit caller-declared physical layout arithmetic, not source truth or runtime."""

import argparse
import json


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def has_evidence(value):
    return isinstance(value, str) and bool(value.strip())


def audit(config):
    missing, conflicts = [], []
    source = config.get("source", {})
    if source.get("source_trace_complete") is not True or not has_evidence(source.get("evidence")):
        missing.append("allocator source trace and active-path evidence")
    if config.get("request_mapping_complete") is not True:
        missing.append("complete request-to-physical-unit mapping")
    if not has_evidence(config.get("layout_identity")):
        missing.append("layout identity (model/backend/hardware/config scope)")

    components = config.get("components", [])
    if not components or len({c["name"] for c in components}) != len(components):
        raise ValueError("components must be nonempty with unique physical-owner names")
    page_bytes = sum(integer(c["pages"], "pages") * integer(c["page_bytes"], "page_bytes", 1)
                     for c in components)
    if config.get("physical_owners_deduplicated") is not True:
        missing.append("physical-owner alias deduplication")

    kind = config["allocator_kind"]
    unit, candidate = None, None
    if kind == "shared_tuple":
        buckets = config["page_buckets_bytes"]
        if not buckets or len(set(buckets)) != len(buckets):
            raise ValueError("page_buckets_bytes must be nonempty unique sizes")
        unit = integer(config["tuple_slots"], "tuple_slots", 1) * sum(
            integer(b, "page bucket bytes", 1) for b in buckets)
        if config.get("tuple_formula_scope_confirmed") is not True:
            missing.append("tuple formula scope (including alignment and trailing slots)")
        groups = config["groups"]
        if not groups or len({g["name"] for g in groups}) != len(groups):
            raise ValueError("groups must be nonempty with unique names")
        # Recompute slots from final groups, not original model layer counts.
        geometry = []
        if not has_evidence(config.get("grouping_evidence")):
            missing.append("backend grouping replay evidence (inputs, outputs and pinned source)")
        for group in groups:
            counts = group.get("owners_per_bucket")
            if not isinstance(counts, list) or len(counts) != len(buckets):
                missing.append(f"group {group['name']}: final owners_per_bucket")
                continue
            geometry.extend(integer(n, "bucket owner count") for n in counts)
        trailing = config.get("trailing_slots")
        if trailing is None:
            missing.append("explicit trailing_slots (zero when absent)")
        elif geometry and len(geometry) == len(groups) * len(buckets):
            derived_slots = max(geometry) + integer(trailing, "trailing_slots")
            if derived_slots != config["tuple_slots"]:
                conflicts.append("tuple_slots differs from final group/bucket geometry plus trailing slots")
        blocks = sum(integer(g["blocks"], "group blocks") for g in groups)
        candidate = blocks * unit
        declared = config.get("declared_pool_bytes_per_block")
        if declared is not None and integer(declared, "declared unit", 1) != unit:
            conflicts.append("declared pool unit differs from allocator geometry")
        admission = config.get("admission_unit_bytes")
        if admission is not None and integer(admission, "admission unit", 1) != unit:
            conflicts.append("admission unit differs from allocator unit")
    elif kind == "independent_pages":
        candidate = page_bytes
    else:
        missing.append(f"unsupported allocator mapping: {kind}; model actual pools separately")

    if candidate is not None and candidate < page_bytes:
        conflicts.append("pool demand is smaller than declared unique-owner pages; check mapping/scope")

    log_status, allocated = "not_provided", None
    log = config.get("log_check")
    if log is not None:
        scope_ok = (log.get("snapshot_scope_confirmed") is True
                    and all(has_evidence(log.get(k)) for k in ("rank", "pool", "evidence")))
        if not scope_ok:
            log_status = "unverified_scope"
            missing.append("same-rank/same-pool unique-backing log scope")
        else:
            allocated = integer(log["bytes"], "log bytes")
            if unit is None:
                log_status = "scope_declared_no_uniform_unit_comparison"
            else:
                blocks = integer(log["blocks"], "log blocks", 1)
                if allocated != blocks * unit:
                    log_status = "conflict"
                    conflicts.append("log bytes/blocks differs from allocator unit")
                else:
                    log_status = "arithmetic_consistent_scope_caller_declared"

    complete = not missing and not conflicts
    need = candidate if complete else None
    return {
        "scope": "caller-declared layout arithmetic; no source validation, alias discovery or runtime measurement",
        "layout_completeness": "conflict" if conflicts else "complete_as_declared" if complete else "incomplete",
        "result_kind": "declared_request_pool_need" if complete else "component_page_subtotal_only",
        "payload_bytes": None,
        "component_page_bytes": page_bytes,
        "allocator_unit_bytes": unit,
        "request_pool_need_bytes": need,
        "pool_layout_overhead_bytes": need - page_bytes if need is not None else None,
        "allocated_pool_bytes": allocated,
        "observed_peak_bytes": None,
        "log_check_status": log_status,
        "missing": missing,
        "conflicts": conflicts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    try:
        with open(args.input, encoding="utf-8") as stream:
            result = audit(json.load(stream))
    except (KeyError, TypeError, ValueError, OSError) as error:
        parser.exit(2, f"invalid layout input: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["layout_completeness"] == "complete_as_declared" else 1


if __name__ == "__main__":
    raise SystemExit(main())
