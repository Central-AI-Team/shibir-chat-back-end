"""Split the reviewed eval set into train.json / test.json.

    python -m scripts.split_eval_set [--seed-file scripts/eval/seed.json]
        [--train scripts/eval/train.json] [--test scripts/eval/test.json]
        [--test-share 0.3] [--seed 20261025]

Only records with review == "accepted" are used. The split is 70/30,
stratified by (slice, category), with a fixed seed, so re-running on the
same accepted set gives the same files byte for byte. A Bengali record and
its Banglish/English variants (variant_of) always land on the same side:
otherwise a test question would be seen in training through its
translation. Both files are written sorted by id and must never be edited
by hand; test.json is never used for tuning (scripts/eval/README.md).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

EVAL_DIR = Path(__file__).parent / "eval"
DEFAULT_SEED = 20261025
DEFAULT_TEST_SHARE = 0.30


def families(records: list[dict]) -> list[list[dict]]:
    """Group a bn record with its variants; everything else is a singleton.
    Each family is ordered: primary first, then variants by id."""
    by_id = {r["id"]: r for r in records}
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in sorted(records, key=lambda r: r["id"]):
        parent = r.get("variant_of")
        key = parent if parent and parent in by_id else r["id"]
        groups[key].append(r)
    out = []
    for key in sorted(groups):
        fam = groups[key]
        fam.sort(key=lambda r: (r["id"] != key, r["id"]))
        out.append(fam)
    return out


def stratum(family: list[dict]) -> tuple:
    primary = family[0]
    return (primary.get("slice"), primary.get("category"))


def split(records: list[dict], *, test_share: float = DEFAULT_TEST_SHARE, seed: int = DEFAULT_SEED
          ) -> tuple[list[dict], list[dict]]:
    """(train, test), each sorted by id. Deterministic for a given input."""
    accepted = [r for r in records if r.get("review") == "accepted"]
    strata: dict[tuple, list[list[dict]]] = defaultdict(list)
    for fam in families(accepted):
        strata[stratum(fam)].append(fam)

    rng = random.Random(seed)
    train: list[dict] = []
    test: list[dict] = []
    for key in sorted(strata, key=str):
        fams = strata[key]
        rng.shuffle(fams)
        n = len(fams)
        n_test = int(n * test_share + 0.5) if n >= 2 else 0
        for i, fam in enumerate(fams):
            (test if i < n_test else train).extend(fam)
    train.sort(key=lambda r: r["id"])
    test.sort(key=lambda r: r["id"])
    return train, test


def _dump(path: Path, records: list[dict]) -> None:
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scripts.split_eval_set")
    ap.add_argument("--seed-file", type=Path, default=EVAL_DIR / "seed.json")
    ap.add_argument("--train", type=Path, default=EVAL_DIR / "train.json")
    ap.add_argument("--test", type=Path, default=EVAL_DIR / "test.json")
    ap.add_argument("--test-share", type=float, default=DEFAULT_TEST_SHARE)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args(argv)

    records = json.loads(args.seed_file.read_text(encoding="utf-8"))
    train, test = split(records, test_share=args.test_share, seed=args.seed)
    if not train and not test:
        print(f"no accepted records in {args.seed_file}; nothing written")
        return 1
    args.train.parent.mkdir(parents=True, exist_ok=True)
    _dump(args.train, train)
    _dump(args.test, test)

    n_acc = len(train) + len(test)
    print(f"{len(records)} records, {n_acc} accepted -> train {len(train)}, test {len(test)} "
          f"(seed {args.seed}, test share {args.test_share})")
    tr, te = Counter(r["slice"] for r in train), Counter(r["slice"] for r in test)
    for s in sorted(set(tr) | set(te)):
        print(f"  {s:<22} train {tr.get(s, 0):>3}  test {te.get(s, 0):>3}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
