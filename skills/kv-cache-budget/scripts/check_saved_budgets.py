#!/usr/bin/env python3
"""CI entry point: only upload generated reports when every saved ledger releases."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from release_budget import atomic_write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("budget-reports"))
    parser.add_argument("--output-dir", type=Path, default=Path(".budget-gate-output"))
    args = parser.parse_args()
    root = args.root.resolve()
    results = []
    for ledger in sorted(args.root.rglob("ledger.json")):
        if not ledger.resolve().is_relative_to(root):
            results.append(dict(ledger=str(ledger), status="blocked", error="ledger symlink escapes root"))
            continue
        relative = ledger.relative_to(args.root).parent
        review = ledger.with_name("review.json")
        command = [sys.executable, str(Path(__file__).with_name("release_budget.py")),
                   "--input", str(ledger), "--review", str(review),
                   "--output-dir", str(args.output_dir / relative)]
        completed = subprocess.run(command, capture_output=True, text=True)
        results.append(dict(ledger=str(ledger), status="released" if completed.returncode == 0 else "blocked",
                            output=completed.stdout.strip(), error=completed.stderr.strip()))
    passed = all(result["status"] == "released" for result in results)
    index = dict(status="passed" if passed else "blocked", checked=len(results), reports=results)
    atomic_write(args.output_dir / "index.json", json.dumps(index, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(index, ensure_ascii=False, indent=2))
    if not results:
        print("No saved ledgers; no deployment report has been certified.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
