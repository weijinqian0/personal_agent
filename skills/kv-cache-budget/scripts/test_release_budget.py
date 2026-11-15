"""Exercise release behavior, not just the calculator's arithmetic."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from release_budget import CHECKS, LIFECYCLE, check_ledger, release, render


def fixture(root):
    """Synthetic hybrid fixture; this is not evidence of a real deployment."""
    source = b"synthetic source: full-history + paged state + fixed draft state\n"
    config = b"synthetic resolved configuration for release-gate tests\n"
    (root / "source.txt").write_bytes(source)
    (root / "config.txt").write_bytes(config)

    def assertion():
        return dict(status="verified", reason="Synthetic test assertion", evidence_ids=["src"])

    values = dict(length=32, prefix_alignment=4, block_size=2, tp=1, dp=1, pp=1, pcp=1, dcp=1,
                  max_num_batched_tokens=4, max_concurrent_batches=2, async_scheduling=True,
                  prefix_cache_enabled=True, speculative_method="synthetic", num_speculative_tokens=1,
                  kv_dtype="synthetic")

    def part(records, pages, na=False):
        return dict(status="not_applicable" if na else "known", records=records, pages=pages,
                    formula=f"declared records={records}, uniquely attributed pages={pages}",
                    reason="Synthetic lifecycle attribution", evidence_ids=["src"])

    inventory = [dict(id="main", owner="target.main", kind="kv", owner_count=2, role="target"),
                 dict(id="state", owner="target.compressor", kind="compressor_state", owner_count=2, role="target"),
                 dict(id="draft", owner="draft.state", kind="fixed_state", owner_count=1, role="draft")]
    shapes = [("full_history", 8, 16, [part(2, 1), part(1, 1), part(1, 0), part(1, 1), part(0, 0), part(0, 0)]),
              ("window", 4, 8, [part(1, 1), part(0, 0), part(4, 2), part(4, 2), part(0, 0), part(0, 0)]),
              ("fixed_state", 32, 64, [part(0, 0, True), part(0, 0, True), part(0, 0, True),
                                      part(0, 0, True), part(1, 1), part(0, 0)])]
    components = []
    for owner, (retention, record_bytes, page_bytes, parts) in zip(inventory, shapes):
        component = dict(owner, retention=retention, record_bytes=record_bytes, page_bytes=page_bytes,
                         attention_type="synthetic hybrid", compression_ratio=4,
                         lifecycle={key: assertion() for key in LIFECYCLE},
                         requests=[dict(id="r1", parts=dict(zip(
                             ["prefix", "settled", "chunk:c1", "chunk:c2", "window_state", "extra"], parts)))])
        components.append(component)
    layout = dict(allocator_kind="independent_pages", layout_identity="synthetic test snapshot",
                  source=dict(source_trace_complete=True, evidence="source.txt"),
                  request_mapping_complete=True, physical_owners_deduplicated=True,
                  components=[dict(name="main", pages=6, page_bytes=16),
                              dict(name="state", pages=10, page_bytes=8),
                              dict(name="draft", pages=1, page_bytes=64)])
    return dict(schema_version=1,
                identity=dict(model="synthetic", model_revision="fixture-v1", hardware="synthetic",
                              role="standalone", rank="0", stage="prefill", analyst="fixture-author",
                              analysis_kind="conditional", backends={"toy": "fixture-v1"}, scope_id="snapshot-1"),
                evidence=[dict(id="src", path="source.txt", sha256=hashlib.sha256(source).hexdigest(),
                               line_start=1, line_end=1, kind="source", backend="toy", revision="fixture-v1",
                               claim="Synthetic fixture; never use for deployment"),
                          dict(id="cfg", path="config.txt", sha256=hashlib.sha256(config).hexdigest(),
                               line_start=1, line_end=1, kind="config", claim="Synthetic parameters")],
                parameters={key: dict(status="confirmed", value=value, evidence_ids=["cfg"])
                            for key, value in values.items()},
                checks={key: assertion() for key in CHECKS}, inventory=inventory, components=components,
                requests=[dict(id="r1", prefix_tokens=8, settled_tokens=4, reference_length=32,
                               generated_tokens=0, evidence_ids=["cfg"],
                               chunks=[dict(id="c1", batch_id="b1", tokens=4),
                                       dict(id="c2", batch_id="b2", tokens=4)])],
                layout=layout, cost_checks={key: dict(scope="snapshot-1", bytes=240, evidence_ids=["src"])
                                           for key in ("allocator", "admission", "capacity")})


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = fixture(self.root)
        self.input = self.root / "ledger.json"
        self.review = self.root / "review.json"
        self.save()
        self.approve()

    def save(self):
        self.input.write_text(json.dumps(self.data), encoding="utf-8")

    def approve(self, reviewer="independent-fixture-reviewer"):
        self.review.write_text(json.dumps(dict(
            ledger_sha256=hashlib.sha256(self.input.read_bytes()).hexdigest(),
            evidence_hashes={entry["id"]: entry["sha256"] for entry in self.data["evidence"]},
            reviewer=reviewer, decision="approved", notes="Synthetic regression review, not runtime proof")))

    def result(self):
        self.save()
        return release(self.input, self.review)

    def assert_blocked(self):
        result = self.result()
        self.assertEqual(result["status"], "blocked")
        self.assertIsNone(result["request_pool_need_bytes"])
        self.assertNotIn("physical", result)
        self.assertIn("尚未算完", render(result))
        return result

    def test_complete_reviewed_hybrid_releases_computed_report(self):
        result = self.result()
        self.assertEqual(result["status"], "released")
        self.assertEqual(result["request_pool_need_bytes"], 240)
        self.assertIsNone(result["observed_peak_bytes"])
        self.assertEqual(len(result["rows"]), 18)
        self.assertIn("chunk:c2", render(result))

    def test_missing_review_blocks_even_complete_arithmetic(self):
        self.assertEqual(release(self.input)["status"], "blocked")

    def test_missing_each_component_lifecycle_operation_blocks(self):
        original = copy.deepcopy(self.data)
        for operation in LIFECYCLE:
            self.data = copy.deepcopy(original)
            del self.data["components"][1]["lifecycle"][operation]
            self.save()
            self.approve()
            with self.subTest(operation=operation):
                self.assert_blocked()

    def test_omitted_state_and_unlisted_draft_are_rejected(self):
        self.data["components"].pop(1)
        self.save()
        self.approve()
        self.assert_blocked()

    def test_missing_q_or_second_chunk_or_extra_cannot_be_hidden_by_total(self):
        original = copy.deepcopy(self.data)
        for name in ("settled", "chunk:c2", "extra", "window_state"):
            self.data = copy.deepcopy(original)
            del self.data["components"][0]["requests"][0]["parts"][name]
            self.save()
            self.approve()
            with self.subTest(part=name):
                self.assert_blocked()

    def test_unknown_is_not_zero_and_cannot_release(self):
        part = self.data["components"][1]["requests"][0]["parts"]["prefix"]
        part.update(status="unknown", records=None, pages=None)
        self.save()
        self.approve()
        self.assert_blocked()

    def test_full_history_prefix_q_chunks_conserve_cumulative_records(self):
        self.data["components"][0]["requests"][0]["parts"]["chunk:c2"]["records"] = 0
        self.assertTrue(any("conservation" in error for error in self.assert_blocked()["errors"]))

    def test_fixed_state_is_not_multiplied_by_chunk_tokens(self):
        result = self.result()
        draft = [row for row in result["rows"] if row["component"] == "draft"]
        self.assertEqual(sum(row["page_bytes"] for row in draft), 64)
        self.assertEqual(sum(row["payload_bytes"] for row in draft), 32)

    def test_each_shared_batch_budget_is_checked_across_requests(self):
        request = copy.deepcopy(self.data["requests"][0])
        request["id"] = "r2"
        self.data["requests"].append(request)
        result = self.assert_blocked()
        self.assertTrue(any("shared token budget exceeded" in error for error in result["errors"]))

    def test_unavailable_tokens_or_excess_batches_block(self):
        self.data["requests"][0]["chunks"][1]["tokens"] = 40
        self.assertTrue(any("available history" in error for error in self.assert_blocked()["errors"]))
        self.data = fixture(self.root)
        self.data["parameters"]["max_concurrent_batches"]["value"] = 1
        self.assert_blocked()

    def test_source_edit_or_revision_mismatch_blocks(self):
        (self.root / "source.txt").write_text("changed after review\n")
        self.assert_blocked()
        self.data = fixture(self.root)
        self.data["evidence"][0]["revision"] = "different-checkout"
        self.assert_blocked()

    def test_source_range_path_escape_and_symlink_escape_block(self):
        original = copy.deepcopy(self.data)
        for path, end in (("../outside.txt", 1), ("source.txt", 100)):
            self.data = copy.deepcopy(original)
            self.data["evidence"][0].update(path=path, line_end=end)
            with self.subTest(path=path):
                self.assert_blocked()
        (self.root / "escape").symlink_to(self.root.parent, target_is_directory=True)
        self.data = original
        self.data["evidence"][0]["path"] = "escape/outside.txt"
        self.assert_blocked()

    def test_stale_review_even_if_new_ledger_is_structurally_valid(self):
        self.data["identity"]["stage"] = "different-phase"
        self.assertTrue(any("stale" in error for error in self.assert_blocked()["errors"]))

    def test_reviewer_must_be_distinct_and_approve(self):
        self.approve(reviewer="fixture-author")
        self.assert_blocked()
        self.approve()
        review = json.loads(self.review.read_text())
        review["decision"] = "rejected"
        self.review.write_text(json.dumps(review))
        self.assert_blocked()

    def test_capacity_conflict_and_missing_cost_evidence_block(self):
        self.data["cost_checks"]["capacity"]["bytes"] = 220
        self.save()
        self.approve()
        self.assertTrue(any("physical cost conflict" in error for error in self.assert_blocked()["errors"]))
        self.data = fixture(self.root)
        del self.data["cost_checks"]["capacity"]["evidence_ids"]
        self.assert_blocked()

    def test_unsafe_flags_do_not_replace_evidence_or_page_mapping(self):
        self.data["checks"]["active_path"] = dict(status="verified", reason="trust me", evidence_ids=[])
        self.save()
        self.approve()
        self.assert_blocked()
        self.data = fixture(self.root)
        self.data["layout"]["components"][0]["pages"] += 1
        self.assert_blocked()

    def test_assumptions_cannot_be_relabelled_as_actual_deployment(self):
        self.data["identity"]["analysis_kind"] = "deployment"
        self.data["parameters"]["max_concurrent_batches"]["status"] = "assumed"
        self.save()
        self.approve()
        self.assert_blocked()

    def test_bad_integer_types_do_not_coerce_to_zero(self):
        for value in (True, -1, 0.5):
            data = copy.deepcopy(self.data)
            data["components"][0]["requests"][0]["parts"]["prefix"]["pages"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                check_ledger(data, self.root)

    def test_cli_failed_rerun_replaces_successful_artifacts(self):
        output = self.root / "out"
        command = [sys.executable, str(Path(__file__).with_name("release_budget.py")),
                   "--input", str(self.input), "--review", str(self.review), "--output-dir", str(output)]
        passed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(passed.returncode, 0, passed.stdout + passed.stderr)
        self.input.write_text("broken json")
        failed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(failed.returncode, 2)
        result = json.loads((output / "result.json").read_text())
        self.assertIsNone(result["request_pool_need_bytes"])
        report = (output / "report.md").read_bytes()
        self.assertEqual(result["report_sha256"], hashlib.sha256(report).hexdigest())
        self.assertIn("尚未算完", report.decode())

    def test_each_pinned_backend_needs_its_own_source_fingerprint(self):
        self.data["identity"]["backends"]["unread-backend"] = "pinned-v2"
        self.save()
        self.approve()
        self.assertTrue(any("no fingerprint-verified source" in error for error in self.assert_blocked()["errors"]))

    def test_floating_versions_and_duplicate_physical_owners_block(self):
        self.data["identity"]["model_revision"] = "main"
        self.assert_blocked()
        self.data = fixture(self.root)
        self.data["inventory"][1]["owner"] = self.data["inventory"][0]["owner"]
        self.data["components"][1]["owner"] = self.data["components"][0]["owner"]
        self.save()
        self.approve()
        self.assert_blocked()

    def test_cli_refuses_to_overwrite_source_or_review(self):
        evidence_bytes = (self.root / "source.txt").read_bytes()
        # Deliberately choose an evidence filename that the renderer would write.
        (self.root / "report.md").write_bytes(evidence_bytes)
        self.data["evidence"][0]["path"] = "report.md"
        self.save()
        self.approve()
        command = [sys.executable, str(Path(__file__).with_name("release_budget.py")),
                   "--input", str(self.input), "--review", str(self.review),
                   "--output-dir", str(self.root)]
        completed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 2)
        self.assertEqual((self.root / "report.md").read_bytes(), evidence_bytes)

    def test_template_is_blocked_and_has_no_candidate_total(self):
        template = Path(__file__).parents[1] / "references/ledger-template.json"
        result = release(template)
        self.assertEqual(result["status"], "blocked")
        self.assertIsNone(result["request_pool_need_bytes"])

    def test_saved_budget_ci_accepts_valid_review_then_rejects_stale_review(self):
        command = [sys.executable, str(Path(__file__).with_name("check_saved_budgets.py")),
                   "--root", str(self.root), "--output-dir", str(self.root / "out")]
        first = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        self.data["identity"]["stage"] = "changed"
        self.save()
        second = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(second.returncode, 1)
        self.assertEqual(json.loads((self.root / "out/index.json").read_text())["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
