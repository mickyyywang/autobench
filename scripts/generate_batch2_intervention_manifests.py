#!/usr/bin/env python3
"""Generate the fixed second-batch closed-loop intervention manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_saved_intervention_manifests import (
    _load_episode,
    build_manifest,
    direct_manifest_source,
    load_direct_episode,
    validate_manifest_semantics,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
BATCH2_EPISODE_IDS = (
    *(f"化妆品收纳B_{index}" for index in range(11, 17)),
    "整理办公桌面B_2",
    *(f"整理餐桌A_{index}" for index in range(9, 13)),
    *(f"整理玩具A_{index}" for index in range(1, 6)),
    *(f"整理玩具B_{index}" for index in range(1, 6)),
)
SAVED_EPISODE_IDS = {f"整理餐桌A_{index}" for index in range(9, 13)}


def _latest_aligned(saved_dir: Path, episode_id: str) -> Path:
    candidates = sorted(saved_dir.glob(f"{episode_id}_*__aligned_*.jsonl"))
    if not candidates:
        raise ValueError(f"{episode_id}: no aligned saved episode in {saved_dir}")
    return candidates[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "exp" / "intervention_manifests_2",
    )
    parser.add_argument("--saved-dir", type=Path, default=PROJECT_DIR / "saved")
    parser.add_argument("--view-graph-dir", type=Path, default=PROJECT_DIR / "view_graph")
    parser.add_argument("--tasks-dir", type=Path, default=PROJECT_DIR / "outputs")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    validated = 0
    for episode_id in BATCH2_EPISODE_IDS:
        target = args.output_dir / f"{episode_id}_intervention_manifest.json"
        if target.exists() and not args.overwrite:
            raise ValueError(f"manifest already exists (use --overwrite): {target}")
        if episode_id in SAVED_EPISODE_IDS:
            source = _latest_aligned(args.saved_dir, episode_id)
            episode = _load_episode(source)
            manifest = build_manifest(
                source,
                episode,
                add_object_mode="inherit",
            )
            source_label = source.name
        else:
            view_graph = args.view_graph_dir / f"{episode_id}.jsonl"
            tasks = args.tasks_dir / f"{episode_id}_tasks.jsonl"
            episode = load_direct_episode(
                view_graph_path=view_graph,
                tasks_path=tasks,
                episode_id=episode_id,
            )
            manifest = build_manifest(
                None,
                episode,
                manifest_source=direct_manifest_source(
                    view_graph_path=view_graph,
                    tasks_path=tasks,
                    episode=episode,
                ),
                add_object_mode=(
                    "task_collection" if episode_id.startswith("整理玩具") else "inherit"
                ),
            )
            source_label = f"{view_graph.name} + {tasks.name}"
        if not args.no_validate:
            validate_manifest_semantics(episode, manifest)
            validated += 1
        target.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        enabled_add = [
            condition["condition_id"]
            for condition in manifest["conditions"]
            if condition.get("eligible") is True
            and condition["condition_id"].startswith("add_object_")
        ]
        if len(enabled_add) != 1:
            raise ValueError(
                f"{episode_id}: expected one eligible add-object condition, got {enabled_add}"
            )
        print(
            f"WROTE {target.name}: source={source_label}; add_object={enabled_add[0]}"
        )
        written += 1
    print(f"Generated {written} batch-2 manifests; semantically validated {validated}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
