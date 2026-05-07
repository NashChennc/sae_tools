from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sae_tools.workflow.artifacts import artifact_meta_path, build_artifact_meta, mark_done, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect statistical top features for bounded geometric analysis.")
    parser.add_argument("--stat-dirs", nargs="+", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed-limit", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _collect_from_stat_dir(path: Path) -> list[int]:
    seeds: list[int] = []
    top_features = _read_json(path / "top_features.json")
    for records in top_features.values():
        for record in records:
            if "feature" in record:
                seeds.append(int(record["feature"]))
    pareto = _read_json(path / "pareto_front.json")
    for feature_id in pareto.get("feature_ids", []):
        seeds.append(int(feature_id))
    return seeds


def main() -> None:
    args = parse_args()
    if args.out.exists() and not args.overwrite:
        print(f"READY {args.out}")
        return

    ordered: list[int] = []
    seen: set[int] = set()
    inputs = [Path(item) for item in args.stat_dirs]
    for stat_dir in inputs:
        for feature_id in _collect_from_stat_dir(stat_dir):
            if feature_id in seen:
                continue
            seen.add(feature_id)
            ordered.append(feature_id)
            if len(ordered) >= args.seed_limit:
                break
        if len(ordered) >= args.seed_limit:
            break

    payload = {
        "seed_limit": args.seed_limit,
        "n_seeds": len(ordered),
        "features": ordered,
        "stat_dirs": [str(path) for path in inputs],
    }
    write_json_atomic(args.out, payload)
    write_json_atomic(
        artifact_meta_path(args.out),
        build_artifact_meta(
            script="collect_stat_seeds.py",
            repo_dir=REPO_ROOT,
            params={"seed_limit": args.seed_limit, "output": str(args.out)},
            inputs={"stat_dirs": [str(path) for path in inputs]},
        ),
    )
    mark_done(args.out)
    print(f"DONE {args.out}")


if __name__ == "__main__":
    main()
