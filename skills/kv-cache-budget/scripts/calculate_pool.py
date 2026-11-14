#!/usr/bin/env python3
"""Recompute a declared additive KV pool budget; no framework dependencies."""

import argparse
import json
import sys


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def ceil_div(n, d):
    return (n + d - 1) // d


def calculate(config):
    unit = integer(config["pool_bytes_per_block"], "pool_bytes_per_block", 1)
    maximum = integer(config["max_model_len"], "max_model_len", 1)
    groups = config["groups"]
    requests = config["requests"]
    if not groups or not requests:
        raise ValueError("groups and requests must be nonempty")
    if len({g["name"] for g in groups}) != len(groups):
        raise ValueError("group names must be unique")
    if len({r["id"] for r in requests}) != len(requests):
        raise ValueError("request ids must be unique")
    in_flight = sum(integer(r["in_flight_tokens"], "in_flight_tokens") for r in requests)
    if "global_in_flight_limit" in config:
        limit = integer(config["global_in_flight_limit"], "global_in_flight_limit")
        if in_flight > limit:
            raise ValueError("sum of request in-flight tokens exceeds global limit")
    results = []
    for request in requests:
        length = integer(request["history_tokens"], "history_tokens")
        flight = integer(request["in_flight_tokens"], "in_flight_tokens")
        if length > maximum or flight > length:
            raise ValueError("require in_flight_tokens <= history_tokens <= max_model_len")
        breakdown = []
        for group in groups:
            multiplicity = integer(group.get("multiplicity", 1), "multiplicity", 1)
            kind = group["kind"]
            if kind == "fixed":
                blocks = integer(group["blocks"], "blocks")
            else:
                block = integer(group["block_tokens"], "block_tokens", 1)
                if kind == "full":
                    blocks = ceil_div(length, block)
                elif kind == "window":
                    window = integer(group["window"], "window", 1)
                    extra = integer(group.get("extra_retained_tokens", 0), "extra_retained_tokens")
                    slack = integer(group["boundary_slack_blocks"], "boundary_slack_blocks")
                    tokens = min(maximum, window - 1 + extra + flight)
                    blocks = ceil_div(tokens, block) + slack
                else:
                    raise ValueError(f"unsupported group kind: {kind}")
            blocks *= multiplicity
            breakdown.append({"group": group["name"], "blocks": blocks, "bytes": blocks * unit})
        total_blocks = sum(g["blocks"] for g in breakdown)
        total_bytes = total_blocks * unit
        results.append({"id": request["id"], "groups": breakdown,
                        "blocks": total_blocks, "bytes": total_bytes,
                        "GB": total_bytes / 10**9, "GiB": total_bytes / 2**30})
    total = sum(r["bytes"] for r in results)
    return {"scope": "declared additive pool budget; no cross-request deduplication or runtime measurement",
            "requests": results, "total_in_flight_tokens": in_flight,
            "total_bytes": total, "total_GB": total / 10**9, "total_GiB": total / 2**30}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSON layout and request scenarios")
    args = parser.parse_args()
    try:
        with open(args.input, encoding="utf-8") as stream:
            result = calculate(json.load(stream))
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, f"invalid budget input: {error}\n")
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    print()


if __name__ == "__main__":
    main()
