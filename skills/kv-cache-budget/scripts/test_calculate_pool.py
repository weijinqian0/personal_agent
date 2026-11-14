"""Regression checks against supplied log arithmetic and budget invariants."""

import copy
import unittest

from calculate_pool import calculate


def scenario(flight=8192, length=131072):
    return {
        "pool_bytes_per_block": 592000, "max_model_len": 1048576,
        "global_in_flight_limit": 32768,
        "groups": [
            {"name": "c4", "kind": "full", "block_tokens": 128},
            {"name": "c128", "kind": "full", "block_tokens": 4096},
            {"name": "swa", "kind": "window", "block_tokens": 32, "window": 128,
             "multiplicity": 2, "boundary_slack_blocks": 1},
            {"name": "c4state", "kind": "window", "block_tokens": 2, "window": 8,
             "boundary_slack_blocks": 1},
            {"name": "c128state", "kind": "window", "block_tokens": 4, "window": 128,
             "boundary_slack_blocks": 1}],
        "requests": [{"id": "A", "history_tokens": length, "in_flight_tokens": flight}]}


class BudgetTests(unittest.TestCase):
    def test_log_scenarios(self):
        for flight, blocks in [(8192, 7760), (16384, 14416), (32768, 27728)]:
            with self.subTest(flight=flight):
                result = calculate(scenario(flight))
                self.assertEqual(result["requests"][0]["blocks"], blocks)
                self.assertEqual(result["total_bytes"], blocks * 592000)

    def test_two_requests_share_global_budget(self):
        config = scenario(16384)
        other = copy.deepcopy(config["requests"][0])
        other["id"] = "B"
        config["requests"].append(other)
        self.assertEqual(calculate(config)["total_bytes"], 17068544000)
        config["requests"][1]["in_flight_tokens"] += 1
        with self.assertRaises(ValueError):
            calculate(config)

    def test_capacity_log_integer_rounding(self):
        blocks = calculate(scenario(32768, 1048576))["requests"][0]["blocks"]
        self.assertEqual(blocks, 35120)
        self.assertEqual(43869 * 1048576 // blocks, 1309794)

    def test_window_cap_and_retention(self):
        config = scenario(10, 10)
        config["max_model_len"] = 10
        config["groups"] = [{"name": "state", "kind": "window", "block_tokens": 4,
                             "window": 8, "extra_retained_tokens": 5,
                             "boundary_slack_blocks": 1}]
        self.assertEqual(calculate(config)["requests"][0]["blocks"], 4)

    def test_fixed_ring_and_non_aligned_full_page(self):
        config = scenario(1, 129)
        config["groups"] = [config["groups"][0], {"name": "ring", "kind": "fixed", "blocks": 1}]
        self.assertEqual(calculate(config)["requests"][0]["blocks"], 3)

    def test_reject_invalid_geometry(self):
        for value in (0, -1, True, 1.5):
            config = scenario()
            config["groups"][0]["block_tokens"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                calculate(config)
        config = scenario(13109, 13108)
        with self.assertRaises(ValueError):
            calculate(config)

    def test_async_fixed_settled_history_adds_long_and_window_pages(self):
        # Declared geometry only: no runtime or automatic model-layout inference.
        config = {
            "pool_bytes_per_block": 462848, "max_model_len": 1048576,
            "global_in_flight_limit": 16384,
            "groups": [
                {"name": "long", "kind": "full", "block_tokens": 128},
                {"name": "ring", "kind": "fixed", "blocks": 1},
                {"name": "swa", "kind": "window", "block_tokens": 128,
                 "window": 128, "multiplicity": 10, "boundary_slack_blocks": 1}],
            "requests": [{"id": "A", "history_tokens": 117888 + 8192,
                          "in_flight_tokens": 8192}]}
        sync = calculate(config)["requests"][0]
        config["requests"][0].update(history_tokens=117888 + 16384,
                                     in_flight_tokens=16384)
        async_result = calculate(config)["requests"][0]
        before = {g["group"]: g["blocks"] for g in sync["groups"]}
        after = {g["group"]: g["blocks"] for g in async_result["groups"]}
        self.assertEqual(after["long"] - before["long"], 64)
        self.assertEqual(after["swa"] - before["swa"], 640)
        self.assertEqual(after["ring"], before["ring"])
        self.assertEqual(async_result["bytes"] - sync["bytes"], 704 * 462848)

        # At fixed complete H, F changes the partition, not the long-KV total.
        config["requests"][0]["in_flight_tokens"] = 8192
        same_history = calculate(config)["requests"][0]
        same_groups = {g["group"]: g["blocks"] for g in same_history["groups"]}
        self.assertEqual(same_groups["long"], after["long"])
        self.assertEqual(same_groups["ring"], after["ring"])
        self.assertEqual(async_result["bytes"] - same_history["bytes"], 640 * 462848)


if __name__ == "__main__":
    unittest.main()
