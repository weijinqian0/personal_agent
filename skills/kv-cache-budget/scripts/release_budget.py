#!/usr/bin/env python3
"""Fail-closed KV ledger/report gate; evidence semantics still require review."""

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import tempfile

from audit_layout import audit


PARAMETERS = {
    "length", "prefix_alignment", "block_size",
    "max_num_batched_tokens", "max_concurrent_batches", "async_scheduling",
    "prefix_cache_enabled", "speculative_method", "num_speculative_tokens", "kv_dtype",
}
CHECKS = {
    "backend_compatibility", "active_path", "inventory", "lifecycle", "allocator",
    "admission", "capacity", "prefix_retention", "speculation", "async",
}
LIFECYCLE = {"write", "restore", "retain", "release", "deduplicate"}
RETENTIONS = {"full_history", "window", "fixed_state"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name}: expected integer >= {minimum}")
    return value


def mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name}: expected object")
    return value


def items(value, name):
    if not isinstance(value, list):
        raise ValueError(f"{name}: expected array")
    return value


def nonempty(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}: expected nonempty string")
    return value


def indexed(value, name):
    result = {}
    for entry in items(value, name):
        entry = mapping(entry, name)
        key = nonempty(entry.get("id"), f"{name}.id")
        if key in result:
            raise ValueError(f"{name}: duplicate id {key}")
        result[key] = entry
    return result


def check_ledger(ledger, root):
    """Verify structure, source fingerprints, accounting and declared evidence links.

    A reference to source is not proof that the caller interpreted it correctly.
    An independent, input-bound review is checked separately before release.
    """
    ledger = mapping(ledger, "ledger")
    errors, rows, evidence_hashes = [], [], {}
    if type(ledger.get("schema_version")) is not int or ledger.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    identity = mapping(ledger.get("identity", {}), "identity")
    for key in ("model", "model_revision", "hardware", "stage", "analyst"):
        if not isinstance(identity.get(key), str) or not identity[key].strip():
            errors.append(f"identity.{key} is missing")
    analysis_kind = identity.get("analysis_kind")
    if analysis_kind not in ("deployment", "conditional"):
        errors.append("identity.analysis_kind must be deployment or conditional")
    backends = mapping(identity.get("backends", {}), "identity.backends")
    if not backends or any(not isinstance(v, str) or not v.strip() for v in backends.values()):
        errors.append("pinned identity.backends revisions are required")
    for revision in [identity.get("model_revision")] + list(backends.values()):
        if isinstance(revision, str) and revision.lower() in {"main", "master", "head", "latest"}:
            errors.append("floating model/backend revisions are not pinned evidence")

    evidence = indexed(ledger.get("evidence", []), "evidence")
    root = Path(root).resolve()
    for key, entry in evidence.items():
        try:
            relative = Path(nonempty(entry.get("path"), "evidence.path"))
            if relative.is_absolute():
                raise ValueError("evidence paths must be relative to the ledger directory")
            path = (root / relative).resolve()
            if not path.is_relative_to(root):
                raise ValueError("evidence path escapes ledger directory")
            data = path.read_bytes()
            actual_hash = digest(data)
            if entry.get("sha256") != actual_hash:
                raise ValueError("evidence sha256 differs from actual file")
            lines = data.decode("utf-8").splitlines()
            start = integer(entry.get("line_start"), "line_start", 1)
            end = integer(entry.get("line_end"), "line_end", start)
            if end > len(lines):
                raise ValueError("evidence line range exceeds file")
            if entry.get("kind") not in ("source", "config", "log", "assumption"):
                raise ValueError("invalid evidence kind")
            nonempty(entry.get("claim"), "evidence.claim")
            if entry["kind"] == "source":
                repo = nonempty(entry.get("backend"), "source.backend")
                if repo not in backends or entry.get("revision") != backends[repo]:
                    raise ValueError("source revision does not match pinned backend")
            if entry["kind"] == "assumption" and analysis_kind != "conditional":
                raise ValueError("deployment cannot use assumption evidence")
            evidence_hashes[key] = actual_hash
        except (OSError, UnicodeError, ValueError) as error:
            errors.append(f"evidence {key}: {error}")
    for backend in backends:
        if not any(entry.get("kind") == "source" and entry.get("backend") == backend
                   and key in evidence_hashes for key, entry in evidence.items()):
            errors.append(f"backend {backend}: no fingerprint-verified source evidence")

    def refs(value, label, source=False):
        references = items(value, f"{label}.evidence_ids")
        if not references:
            errors.append(f"{label}: evidence_ids is empty")
        for reference in references:
            if not isinstance(reference, str) or reference not in evidence_hashes:
                errors.append(f"{label}: missing or invalid evidence {reference!r}")
        if source and not any(
            isinstance(ref, str) and ref in evidence_hashes and evidence[ref]["kind"] == "source"
            for ref in references
        ):
            errors.append(f"{label}: source evidence is required")

    def attestation(value, label, source=True):
        value = mapping(value, label)
        if value.get("status") != "verified":
            errors.append(f"{label}: not verified ({value.get('status', 'missing')})")
        if not isinstance(value.get("reason"), str) or not value["reason"].strip():
            errors.append(f"{label}: reason is required")
        refs(value.get("evidence_ids", []), label, source)

    checks = mapping(ledger.get("checks", {}), "checks")
    for key in sorted(CHECKS):
        attestation(checks.get(key, {}), f"checks.{key}")

    parameters, assumptions = {}, []
    declared_parameters = mapping(ledger.get("parameters", {}), "parameters")
    for name in sorted(PARAMETERS):
        value = mapping(declared_parameters.get(name, {}), f"parameters.{name}")
        status = value.get("status")
        if status not in ("confirmed", "assumed") or value.get("value") is None:
            errors.append(f"parameters.{name}: unresolved")
            continue
        if status == "assumed":
            assumptions.append(name)
            if analysis_kind != "conditional":
                errors.append(f"parameters.{name}: assumption in deployment result")
        refs(value.get("evidence_ids", []), f"parameters.{name}")
        parameters[name] = value["value"]
    for name in PARAMETERS - {"kv_dtype", "speculative_method", "async_scheduling", "prefix_cache_enabled"}:
        if name in parameters:
            integer(parameters[name], name, 0 if name == "num_speculative_tokens" else 1)
    for name in ("async_scheduling", "prefix_cache_enabled"):
        if name in parameters and type(parameters[name]) is not bool:
            raise ValueError(f"{name}: expected boolean")
    for name in ("kv_dtype", "speculative_method"):
        if name in parameters:
            nonempty(parameters[name], name)

    requests = indexed(ledger.get("requests", []), "requests")
    if len(requests) != 1:
        errors.append("requests: exactly one request snapshot is supported")
    snapshots, batches = {}, {}
    for key, request in requests.items():
        p = integer(request.get("prefix_tokens"), "prefix_tokens")
        q = integer(request.get("settled_tokens"), "settled_tokens")
        limit = integer(request.get("reference_length"), "reference_length", 1)
        generated = integer(request.get("generated_tokens"), "generated_tokens")
        refs(request.get("evidence_ids", []), f"requests.{key}")
        chunks = indexed(request.get("chunks", []), f"requests.{key}.chunks")
        positions, end = {}, p + q
        for chunk_id, chunk in chunks.items():
            count = integer(chunk.get("tokens"), "chunk.tokens", 1)
            batch = nonempty(chunk.get("batch_id"), "chunk.batch_id")
            positions[chunk_id] = (end, end + count)
            end += count
            batches[batch] = batches.get(batch, 0) + count
        if end > limit + generated:
            errors.append(f"request {key}: P+Q+sum(C_i) exceeds available history")
        if "length" in parameters and limit != parameters["length"]:
            errors.append(f"request {key}: reference_length differs from parameters.length")
        if "prefix_alignment" in parameters and p % parameters["prefix_alignment"]:
            errors.append(f"request {key}: prefix is not aligned")
        if parameters.get("prefix_cache_enabled") is False and p:
            errors.append(f"request {key}: nonzero prefix while prefix caching is disabled")
        snapshots[key] = (p, q, positions)
    if "max_num_batched_tokens" in parameters:
        for batch, count in batches.items():
            if count > parameters["max_num_batched_tokens"]:
                errors.append(f"batch {batch}: single-request batch token limit exceeded")
    if "max_concurrent_batches" in parameters and len(batches) > parameters["max_concurrent_batches"]:
        errors.append("number of in-flight batches exceeds resolved concurrency")

    inventory = indexed(ledger.get("inventory", []), "inventory")
    components = indexed(ledger.get("components", []), "components")
    if not inventory or inventory.keys() != components.keys():
        errors.append("inventory and component coverage differ or inventory is empty")
    owners = [entry.get("owner") for entry in inventory.values()]
    if any(not isinstance(owner, str) for owner in owners) or len(set(owners)) != len(owners):
        errors.append("physical owner identities must be unique after alias deduplication")
    page_components = []
    if parameters.get("speculative_method") not in (None, "none") and not any(
        entry.get("role") == "draft" for entry in inventory.values()
    ):
        errors.append("speculation enabled but draft inventory is absent")
    for key, component in components.items():
        declared = inventory.get(key, {})
        for field in ("owner", "kind", "owner_count", "role"):
            if component.get(field) != declared.get(field) or component.get(field) is None:
                errors.append(f"component {key}: inventory {field} mismatch")
        if component.get("role") not in ("target", "draft"):
            errors.append(f"component {key}: role must be target or draft")
        nonempty(component.get("owner"), "component.owner")
        nonempty(component.get("kind"), "component.kind")
        nonempty(component.get("attention_type"), "component.attention_type")
        owners = integer(component.get("owner_count"), "owner_count", 1)
        record_bytes = integer(component.get("record_bytes"), "record_bytes", 1)
        page_bytes = integer(component.get("page_bytes"), "page_bytes", 1)
        retention = component.get("retention")
        if retention not in RETENTIONS:
            errors.append(f"component {key}: unknown retention type")
        lifecycle = mapping(component.get("lifecycle", {}), "lifecycle")
        for operation in sorted(LIFECYCLE):
            attestation(lifecycle.get(operation, {}), f"components.{key}.lifecycle.{operation}")
        entries = indexed(component.get("requests", []), f"component {key}.requests")
        if entries.keys() != requests.keys():
            errors.append(f"component {key}: request coverage differs")
        total_pages = total_payload = 0
        for request_id, entry in entries.items():
            if request_id not in snapshots:
                continue
            p, q, positions = snapshots[request_id]
            parts = mapping(entry.get("parts", {}), "parts")
            expected = {"prefix", "settled", "window_state", "extra"} | {
                f"chunk:{chunk_id}" for chunk_id in positions
            }
            if parts.keys() != expected:
                errors.append(f"component {key}/{request_id}: missing or unexpected P/Q/chunk/state parts")
            entry_records = entry_pages = 0
            for part_id, part in parts.items():
                label = f"components.{key}.{request_id}.{part_id}"
                part = mapping(part, label)
                status = part.get("status")
                if status not in ("known", "not_applicable"):
                    errors.append(f"{label}: unknown accounting part")
                    rows.append(dict(component=key, request=request_id, part=part_id,
                                     records=None, pages=None, payload_bytes=None, page_bytes=None,
                                     formula=part.get("formula", "unknown"), reason=part.get("reason", "unknown"),
                                     evidence_ids=part.get("evidence_ids", [])))
                    continue
                nonempty(part.get("formula"), f"{label}.formula")
                nonempty(part.get("reason"), f"{label}.reason")
                refs(part.get("evidence_ids", []), label)
                records = integer(part.get("records"), "part.records")
                pages = integer(part.get("pages"), "part.pages")
                if status == "not_applicable" and (records or pages):
                    errors.append(f"{label}: not_applicable must have zero records/pages")
                if retention == "full_history":
                    ratio = integer(component.get("compression_ratio"), "compression_ratio", 1)
                    interval = ((0, p) if part_id == "prefix" else (p, p + q)
                                if part_id == "settled" else positions.get(part_id.removeprefix("chunk:")))
                    if interval is not None:
                        expected_records = interval[1] // ratio - interval[0] // ratio
                        if records != expected_records:
                            errors.append(f"{label}: cumulative record conservation failed")
                payload = records * record_bytes * owners
                physical = pages * page_bytes * owners
                entry_records += records
                entry_pages += pages
                rows.append(dict(component=key, request=request_id, part=part_id,
                                 records=records * owners, pages=pages * owners,
                                 payload_bytes=payload, page_bytes=physical, formula=part["formula"],
                                 reason=part["reason"], evidence_ids=part["evidence_ids"]))
            if entry_records * record_bytes > entry_pages * page_bytes:
                errors.append(f"component {key}/{request_id}: payload exceeds attributed physical pages")
            total_pages += entry_pages * owners
            total_payload += entry_records * record_bytes * owners
        page_components.append(dict(name=key, pages=total_pages, page_bytes=page_bytes))

    layout = mapping(ledger.get("layout", {}), "layout")
    layout_rows = items(layout.get("components", []), "layout.components")
    layout_map = {nonempty(mapping(c, "layout component").get("name"), "layout component.name"): c
                  for c in layout_rows}
    if len(layout_map) != len(layout_rows) or layout_map != {c["name"]: c for c in page_components}:
        errors.append("layout components do not match per-component P/Q/chunk/state page ledger")
    physical = None
    if layout_rows:
        physical = audit(layout)
        errors.extend(f"layout: {message}" for message in physical["missing"] + physical["conflicts"])
    else:
        errors.append("physical layout is missing")
    costs = mapping(ledger.get("cost_checks", {}), "cost_checks")
    if physical and physical["request_pool_need_bytes"] is not None:
        for name in ("allocator", "admission", "capacity"):
            cost = mapping(costs.get(name, {}), f"cost_checks.{name}")
            refs(cost.get("evidence_ids", []), f"cost_checks.{name}", source=True)
            if cost.get("scope") != identity.get("scope_id") or not cost.get("scope"):
                errors.append(f"cost_checks.{name}: snapshot scope differs or is missing")
            if integer(cost.get("bytes"), f"cost_checks.{name}.bytes") != physical["request_pool_need_bytes"]:
                errors.append(f"cost_checks.{name}: physical cost conflict")

    geometry = [{key: component.get(key) for key in
                 ("id", "owner", "kind", "role", "owner_count", "attention_type", "retention",
                  "record_bytes", "page_bytes", "compression_ratio", "lifecycle")}
                for component in components.values()]
    return dict(errors=errors, identity=identity, assumptions=assumptions, rows=rows,
                parameters=declared_parameters, evidence=list(evidence.values()), geometry=geometry,
                requests=list(requests.values()), checks=checks,
                evidence_hashes=evidence_hashes, physical=physical,
                component_page_bytes=sum(r["page_bytes"] or 0 for r in rows),
                payload_bytes=sum(r["payload_bytes"] or 0 for r in rows))


def release(input_path, review_path=None):
    input_path = Path(input_path)
    raw = input_path.read_bytes()
    result = check_ledger(json.loads(raw), input_path.parent)
    result["ledger_sha256"] = digest(raw)
    result["review_status"] = "missing"
    if review_path is None:
        result["errors"].append("independent review record is missing")
    else:
        review = mapping(json.loads(Path(review_path).read_bytes()), "review")
        if review.get("ledger_sha256") != result["ledger_sha256"]:
            result["errors"].append("review is stale: ledger hash differs")
        if review.get("evidence_hashes") != result["evidence_hashes"]:
            result["errors"].append("review is stale: evidence fingerprint set differs")
        if (not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip()
                or review.get("reviewer") == result["identity"].get("analyst")):
            result["errors"].append("independent reviewer identity is missing or equals analyst")
        if review.get("decision") != "approved":
            result["errors"].append("independent review did not approve release")
        nonempty(review.get("notes"), "review.notes")
        result["review_status"] = "approved_as_attested" if not result["errors"] else "not_accepted"
    passed = not result["errors"]
    result["status"] = "released" if passed else "blocked"
    result["request_pool_need_bytes"] = result["physical"]["request_pool_need_bytes"] if passed else None
    result["allocated_pool_bytes"] = result["physical"]["allocated_pool_bytes"] if passed else None
    result["observed_peak_bytes"] = None
    if passed:
        ledger = json.loads(raw)
        result["pool_geometry"] = dict(
            allocator_kind=ledger["layout"]["allocator_kind"],
            unit_bytes=result["physical"]["allocator_unit_bytes"],
            groups=ledger["layout"].get("groups", []),
            overhead_bytes=result["physical"]["pool_layout_overhead_bytes"],
            cost_checks=ledger["cost_checks"])
    result["limitations"] = (
        "Fingerprints, completeness and declared arithmetic checked; source interpretation and "
        "reviewer identity are attestations, not authenticated proof. No runtime peak measured."
    )
    # Internal candidate totals must not escape through a blocked public result.
    result.pop("physical")
    return result


def cell(value):
    return html.escape(str(value)).replace("|", "&#124;").replace("\n", " ")


def render(result):
    passed = result["status"] == "released"
    lines = ["# KV Cache 交付报告", "", f"状态：{'已通过受控门禁' if passed else '未完成；禁止交付部署总量'}", "",
             "## 作用域与证据", "", "| 字段 | 值 |", "|---|---|"]
    for key, value in result.get("identity", {}).items():
        lines.append(f"| {cell(key)} | {cell(value)} |")
    lines += [f"| ledger_sha256 | {result.get('ledger_sha256', 'invalid input')} |", "",
              "## 参数", "", "| 参数 | 值 | 状态 | 证据 |", "|---|---|---|---|"]
    for key, parameter in result.get("parameters", {}).items():
        lines.append("| " + " | ".join(cell(value) for value in
                     (key, parameter.get("value"), parameter.get("status"), parameter.get("evidence_ids"))) + " |")
    lines += ["", "## 证据指纹", "", "| ID | 文件及行范围 | 后端版本 | SHA-256 | 声明 |",
              "|---|---|---|---|---|"]
    for source in result.get("evidence", []):
        values = (source["id"], f"{source.get('path')}:{source.get('line_start')}-{source.get('line_end')}",
                  source.get("revision", source.get("kind")), source.get("sha256"), source.get("claim"))
        lines.append("| " + " | ".join(map(cell, values)) + " |")
    lines += ["", "## 每请求快照", "", "| 请求 | P | Q | 各批chunk（含batch ID） | 参考长度 | 生成位置 |",
              "|---|---:|---:|---|---:|---:|"]
    for request in result.get("requests", []):
        values = (request["id"], request["prefix_tokens"], request["settled_tokens"], request["chunks"],
                  request["reference_length"], request["generated_tokens"])
        lines.append("| " + " | ".join(map(cell, values)) + " |")
    lines += ["", "## 组件与状态生命周期", "",
              "| 组件 | owner/数量/Attention | retention / record bytes / page bytes | 操作 | 状态与理由 | 证据 |",
              "|---|---|---|---|---|---|"]
    for component in result.get("geometry", []):
        for operation, fact in component.get("lifecycle", {}).items():
            values = (component["id"], f"{component['owner']} / {component['owner_count']} / {component['attention_type']}",
                      f"{component['retention']} / {component['record_bytes']} / {component['page_bytes']}",
                      operation, f"{fact.get('status')}: {fact.get('reason')}", fact.get("evidence_ids"))
            lines.append("| " + " | ".join(map(cell, values)) + " |")
    lines += ["", "## 逐组件分账", "", "| 组件 | 请求 | P/Q/chunk/state分项 | 记录 | 页 | payload bytes | 组件页面 bytes | 公式 | 证据及归因 |",
              "|---|---|---|---:|---:|---:|---:|---|---|"]
    for row in result.get("rows", []):
        lines.append("| " + " | ".join(cell(row[key]) if row[key] is not None else "未知" for key in
                     ("component", "request", "part", "records", "pages", "payload_bytes", "page_bytes", "formula"))
                     + " | " + cell(f"{row['evidence_ids']}: {row['reason']}") + " |")
    if passed:
        geometry = result["pool_geometry"]
        lines += ["", "## 物理池映射与成本核对", "",
                  f"allocator={cell(geometry['allocator_kind'])}；unit_bytes={cell(geometry['unit_bytes'])}；"
                  f"groups={cell(geometry['groups'])}；跨组件物理填充={cell(geometry['overhead_bytes'])} bytes。", "",
                  "| 路径 | 同一scope | bytes | 证据 |", "|---|---|---:|---|"]
        for name, cost in geometry["cost_checks"].items():
            lines.append("| " + " | ".join(map(cell, (name, cost["scope"], cost["bytes"], cost["evidence_ids"]))) + " |")
    if not passed:
        lines += ["", "## 缺项与冲突", ""] + [f"- {cell(error)}" for error in result["errors"]]
    lines += ["", "## 末尾汇总", "", "| 口径 | bytes |", "|---|---:|",
              f"| 已知组件页面小计（不是部署总量） | {result.get('component_page_bytes', 0)} |"]
    if passed:
        amount = result["request_pool_need_bytes"]
        lines += [f"| {'条件池需求' if result['identity']['analysis_kind'] == 'conditional' else '请求池需求'} | {amount} |",
                  f"| 请求池需求 GiB | {amount / (1 << 30):.9f} |"]
    else:
        lines.append("| 请求物理池需求 | 尚未算完 |")
    lines += ["| 实测峰值 | 未测量 |", "", result.get("limitations", "Input validation failed."), ""]
    if result.get("assumptions"):
        lines += ["假设参数：" + ", ".join(map(cell, result["assumptions"])), ""]
    return "\n".join(lines)


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output_paths = [args.output_dir / name for name in ("result.json", "report.md", "review-request.json")]
    protected = {args.input.resolve()} | ({args.review.resolve()} if args.review else set())
    try:
        original = json.loads(args.input.read_bytes())
        for entry in original.get("evidence", []):
            if isinstance(entry, dict) and isinstance(entry.get("path"), str):
                protected.add((args.input.parent / entry["path"]).resolve())
    except (OSError, ValueError, AttributeError, TypeError):
        pass  # The regular validation path writes an invalid-input report.
    if any(path.resolve() in protected for path in output_paths):
        parser.error("output must not overwrite ledger, review or evidence")
    code = 1
    try:
        result = release(args.input, args.review)
        code = 0 if result["status"] == "released" else 1
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, AttributeError) as error:
        result = dict(status="blocked", request_pool_need_bytes=None, allocated_pool_bytes=None,
                      observed_peak_bytes=None, errors=[f"invalid input: {error}"])
        code = 2
    report = render(result)
    result["report_sha256"] = digest(report.encode("utf-8"))
    # Even a failed rerun replaces prior successful outputs with a blocked report.
    atomic_write(args.output_dir / "report.md", report)
    atomic_write(args.output_dir / "result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    request = dict(ledger_sha256=result.get("ledger_sha256"), evidence_hashes=result.get("evidence_hashes", {}),
                   reviewer=None, decision="pending", notes="", validation_errors=result["errors"])
    atomic_write(args.output_dir / "review-request.json", json.dumps(request, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], errors=result["errors"],
                          ledger_sha256=result.get("ledger_sha256")), ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
