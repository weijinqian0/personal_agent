import unittest

from calculate_peak import calculate


def ledger():
    return {"rank": 0, "phases": ["attention", "moe"], "coverage": {
        "unassessed_components": [], "unassessed_phases": []}, "storages": []}


def storage(ident, amount, phases, category="activation", status="confirmed"):
    return {"id": ident, "bytes": amount, "live_phases": phases,
            "category": category, "status": status}


class PeakTests(unittest.TestCase):
    def test_persistent_storage_is_added_to_max_not_sum_of_sequential_phases(self):
        d = ledger()
        d["storages"] = [storage("pool", 10, ["*"], "communication"),
                         storage("a", 100, ["attention"]), storage("m", 200, ["moe"])]
        self.assertEqual(calculate(d)["complete_ledger_peak_bytes"], 210)

    def test_unknown_graph_pool_keeps_complete_peak_unknown(self):
        d = ledger()
        d["storages"] = [storage("graph", None, ["*"], "graph_pool", "unknown"),
                         storage("a", 100, ["attention"])]
        r = calculate(d)
        self.assertEqual(r["known_subtotal_peak_bytes"], 100)
        self.assertIsNone(r["complete_ledger_peak_bytes"])

    def test_excluded_unknown_weight_and_kv_do_not_contaminate_scope(self):
        d = ledger()
        d["storages"] = [storage("w", None, ["*"], "weights", "unknown"),
                         storage("kv", 9999, ["*"], "kv_cache"),
                         storage("kv-staging", 20, ["attention"])]
        self.assertEqual(calculate(d)["complete_ledger_peak_bytes"], 20)

    def test_unassessed_phase_prevents_false_complete_result(self):
        d = ledger()
        d["coverage"]["unassessed_phases"] = ["mtp"]
        self.assertIsNone(calculate(d)["complete_ledger_peak_bytes"])

    def test_alias_duplicate_is_rejected_instead_of_double_counted(self):
        d = ledger()
        d["storages"] = [storage("backing", 10, ["attention"])] * 2
        with self.assertRaises(ValueError):
            calculate(d)

    def test_bad_bytes_or_unknown_phase_are_rejected(self):
        for amount in (True, -1, 1.5):
            d = ledger()
            d["storages"] = [storage("bad", amount, ["moe"])]
            with self.assertRaises(ValueError):
                calculate(d)
        d = ledger()
        d["storages"] = [storage("bad", 10, ["mispelled"])]
        with self.assertRaises(ValueError):
            calculate(d)


if __name__ == "__main__":
    unittest.main()
