"""Create reproducible story-level train/dev splits for Human and Auto.

First run (writes split files and a manifest of story indices):
    python make_splits.py
Rebuild identical files from the committed manifest on another machine:
    python make_splits.py --from-manifest
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spartqa.data import read_json, write_json
from spartqa.splits import file_sha256, split_stories, split_summary, subset_data

ROOT = Path(__file__).resolve().parent
MANIFEST_VERSION = 1


def write_split_files(name: str, data: dict, dev_indices: list[int], output_dir: Path) -> dict:
    dev_set = set(dev_indices)
    train_indices = [index for index in range(len(data["data"])) if index not in dev_set]
    summaries = {}
    for part, indices in (("train", train_indices), ("dev", dev_indices)):
        subset = subset_data(data, indices)
        write_json(output_dir / f"{name}_{part}.json", subset)
        summaries[part] = split_summary(subset)
    return summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--human", type=Path, default=Path("Data/human_train.json"))
    parser.add_argument("--auto", type=Path, default=Path("Data/auto_train.json"))
    parser.add_argument("--datasets", nargs="+", choices=("human", "auto"), default=["human", "auto"])
    parser.add_argument("--human-dev-ratio", type=float, default=0.2)
    parser.add_argument("--auto-dev-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=100,
                        help="Seeded shuffles tried; the most label-balanced dev set is kept.")
    parser.add_argument("--output-dir", type=Path, default=Path("Data/splits"))
    parser.add_argument("--from-manifest", action="store_true",
                        help="Rebuild split files from the story indices in the existing manifest.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing manifest.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    manifest_path = args.output_dir / "manifest.json"
    sources = {"human": (args.human, args.human_dev_ratio), "auto": (args.auto, args.auto_dev_ratio)}

    if args.from_manifest:
        if not manifest_path.exists():
            parser.error(f"No manifest at {manifest_path}")
        manifest = read_json(manifest_path)
        for name, entry in manifest["datasets"].items():
            source = Path(entry["source"])
            if not source.exists():
                source = ROOT / entry["source"]
            if not source.exists():
                parser.error(f"Source file {entry['source']} for {name} was not found")
            if file_sha256(source) != entry["source_sha256"]:
                parser.error(f"{source} does not match the manifest checksum; the split cannot be reproduced")
            write_split_files(name, read_json(source), entry["dev_story_indices"], args.output_dir)
            print(f"Rebuilt {name} split from manifest ({len(entry['dev_story_indices'])} dev stories)")
        return 0

    if manifest_path.exists() and not args.force:
        parser.error(f"{manifest_path} exists. Use --from-manifest to reproduce it or --force to replace it.")
    manifest = {"format_version": MANIFEST_VERSION, "seed": args.seed, "trials": args.trials,
                "unit": "story", "datasets": {}}
    for name in args.datasets:
        source, ratio = sources[name]
        data = read_json(source)
        _, dev_indices = split_stories(data, dev_ratio=ratio, seed=args.seed, trials=args.trials)
        summaries = write_split_files(name, data, dev_indices, args.output_dir)
        manifest["datasets"][name] = {
            "source": source.as_posix(), "source_sha256": file_sha256(source),
            "total_stories": len(data["data"]), "dev_ratio": ratio,
            "summary": summaries, "dev_story_indices": dev_indices,
        }
        print(json.dumps({name: summaries}, ensure_ascii=False))
    write_json(manifest_path, manifest)
    print(f"Wrote split files and {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
