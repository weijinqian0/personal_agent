"""Physical accounting regressions; evidence flags remain caller assertions."""

import copy
import json
from pathlib import Path
import unittest

from audit_layout import audit


def fixture():
    return json.loads((Path(__file__).parent / "fixtures/shared-tuple-layout.json").read_text())


class LayoutTests(unittest.TestCase):
    def test_component_pages_do_not_replace_shared_pool(self):
        result = audit(fixture())
        self.assertEqual(result["component_page_bytes"], 6667249792)
        self.assertEqual(result["allocator_unit_bytes"], 592000)
        self.assertEqual(result["request_pool_need_bytes"], 8534272000)
        self.assertEqual(result["pool_layout_overhead_bytes"], 1867022208)
        self.assertIsNone(result["observed_peak_bytes"])

    def test_incomplete_mapping_never_returns_deployment_need(self):
        for field in ("request_mapping_complete", "physical_owners_deduplicated", "tuple_formula_scope_confirmed"):
            config = fixture()
            config[field] = False
            with self.subTest(field=field):
                result = audit(config)
                self.assertIsNone(result["request_pool_need_bytes"])
                self.assertEqual(result["component_page_bytes"], 6667249792)
        config = fixture()
        config["source"]["source_trace_complete"] = False
        self.assertIsNone(audit(config)["request_pool_need_bytes"])

    def test_mtp_tail_changes_cost_and_admission_must_agree(self):
        config = fixture()
        config["admission_unit_bytes"] = 23 * (4224 + 19456)
        self.assertEqual(audit(config)["layout_completeness"], "conflict")
        self.assertIsNone(audit(config)["request_pool_need_bytes"])
        config = fixture()
        config["tuple_slots"] = 26
        config["trailing_slots"] = 3
        config.pop("declared_pool_bytes_per_block")
        self.assertEqual(audit(config)["request_pool_need_bytes"], 14416 * 26 * 23680)

    def test_missing_group_geometry_blocks_total(self):
        for field in ("grouping_evidence", "trailing_slots"):
            config = fixture()
            del config[field]
            self.assertIsNone(audit(config)["request_pool_need_bytes"])
        config = fixture()
        del config["groups"][0]["owners_per_bucket"]
        self.assertIsNone(audit(config)["request_pool_need_bytes"])

    def test_final_group_geometry_overrides_original_layer_count(self):
        config = fixture()
        config["trailing_slots"] = 0
        config["groups"] = [dict(name=f"g{i}", blocks=3000, owners_per_bucket=counts)
                            for i, counts in enumerate(
                                [[21, 21], [0, 20], [0, 22], [0, 21], [21, 21], [0, 20]])]
        config.pop("declared_pool_bytes_per_block")
        config["tuple_slots"] = 21
        self.assertIsNone(audit(config)["request_pool_need_bytes"])
        config["tuple_slots"] = 22
        self.assertEqual(audit(config)["allocator_unit_bytes"], 22 * (4224 + 19456))
        self.assertEqual(audit(config)["request_pool_need_bytes"], 18000 * 22 * (4224 + 19456))

    def test_log_scope_and_cost(self):
        config = fixture()
        config["log_check"] = dict(bytes=25970448000, blocks=43869, rank="DP2", pool="KV",
                                   evidence="supplied log", snapshot_scope_confirmed=False)
        result = audit(config)
        self.assertEqual(result["log_check_status"], "unverified_scope")
        self.assertIsNone(result["allocated_pool_bytes"])
        self.assertIsNone(result["request_pool_need_bytes"])
        config["log_check"]["snapshot_scope_confirmed"] = True
        self.assertEqual(audit(config)["allocated_pool_bytes"], 25970448000)
        config["log_check"]["bytes"] += 1
        self.assertEqual(audit(config)["layout_completeness"], "conflict")

    def test_independent_gqa_pages_need_no_shared_unit(self):
        config = dict(allocator_kind="independent_pages", layout_identity="GQA TP1 BF16 B128",
                      source=dict(source_trace_complete=True, evidence="independent allocator trace"),
                      request_mapping_complete=True, physical_owners_deduplicated=True,
                      components=[dict(name="kv", pages=2, page_bytes=128*2*8*128*2)])
        result = audit(config)
        self.assertEqual(result["request_pool_need_bytes"], 2*128*2*8*128*2)
        self.assertIsNone(result["allocator_unit_bytes"])

    def test_unsupported_multiple_pools_and_alias_are_not_silently_summed(self):
        config = fixture()
        config["allocator_kind"] = "multiple_pools"
        self.assertIsNone(audit(config)["request_pool_need_bytes"])
        config = fixture()
        config["components"].append(copy.deepcopy(config["components"][0]))
        with self.assertRaises(ValueError):
            audit(config)

    def test_invalid_unit_geometry(self):
        for bad in (True, 0, -1, 1.5):
            config = fixture()
            config["tuple_slots"] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                audit(config)

    def test_pool_cannot_be_smaller_than_unique_owner_pages(self):
        config = fixture()
        config["groups"] = [dict(name="incomplete", blocks=1)]
        result = audit(config)
        self.assertEqual(result["layout_completeness"], "conflict")
        self.assertIsNone(result["request_pool_need_bytes"])


if __name__ == "__main__":
    unittest.main()
