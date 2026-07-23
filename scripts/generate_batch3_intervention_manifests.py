#!/usr/bin/env python3
"""Generate the fixed third/fourth-batch closed-loop intervention manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_saved_intervention_manifests import (
    build_manifest,
    direct_manifest_source,
    load_direct_episode,
    validate_manifest_semantics,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
BATCH3_EPISODE_IDS = (
    *(f"DivideBuffetTraysA_{index}" for index in range(1, 8)),
    *(f"LoadCondimentsInFridgeA_{index}" for index in range(1, 4)),
)
BATCH4_EPISODE_IDS = tuple(f"LoadCondimentsInFridgeA_{index}" for index in range(4, 11))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, choices=(3, 4), default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--view-graph-dir", type=Path, default=PROJECT_DIR / "view_graph")
    parser.add_argument("--tasks-dir", type=Path, default=PROJECT_DIR / "outputs")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    episode_ids = BATCH3_EPISODE_IDS if args.batch == 3 else BATCH4_EPISODE_IDS
    output_dir = args.output_dir or PROJECT_DIR / "exp" / f"intervention_manifests_{args.batch}"

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    validated = 0
    for episode_id in episode_ids:
        target = output_dir / f"{episode_id}_intervention_manifest.json"
        if target.exists() and not args.overwrite and not args.dry_run:
            raise ValueError(f"manifest already exists (use --overwrite): {target}")

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
            add_object_mode="inherit",
        )
        if not args.no_validate:
            validate_manifest_semantics(episode, manifest)
            validated += 1

        add_condition = next(
            condition
            for condition in manifest["conditions"]
            if condition["condition_id"] == "add_object_inherit_source_goal"
        )
        if add_condition.get("eligible") is not True:
            raise ValueError(f"{episode_id}: inherited add-object condition is not eligible")
        disturbance = add_condition["graph_disturbance"]
        summary = (
            f"copy={disturbance['object']['copy_from']}->{disturbance['object']['id']}; "
            f"spawn={disturbance['relation']}({disturbance['object']['id']},"
            f"{disturbance['target']}); "
            f"goals={disturbance['success_policy']['placement_alternatives']}"
        )
        if args.dry_run:
            print(f"VALID {episode_id}: {summary}")
            continue

        target.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"WROTE {target.name}: {summary}")
        written += 1

    verb = "Checked" if args.dry_run else "Generated"
    count = len(episode_ids) if args.dry_run else written
    print(
        f"{verb} {count} batch-{args.batch} manifests; "
        f"semantically validated {validated}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
