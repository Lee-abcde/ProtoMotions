# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the full eight-rank source manifest for one locomotion rank on Hyperion."""

import argparse
import copy
from pathlib import Path

import yaml


def relocate_manifest(
    template: dict,
    data: Path,
    hoi: Path,
    loco: Path,
    amass_motion_file: Path | None = None,
) -> dict:
    """Keep every source; one locomotion rank, HOI round-robin over the rest."""
    manifest = copy.deepcopy(template)
    manifest["locomotion_num_ranks"] = 1
    roots = {
        "/workspace/protomotion_data": data,
        "/workspace/hoi_teachers": hoi,
        "/workspace/loco_teachers": loco,
    }
    for source in manifest["sources"]:
        for field in ("motion_file", "scenes_file", "teacher_checkpoint"):
            if field not in source:
                continue
            path = Path(source[field])
            for prefix, root in roots.items():
                if path.is_relative_to(prefix):
                    source[field] = str(root / path.relative_to(prefix))
                    break
            else:
                raise ValueError(f"Unrecognized template path: {path}")
        if source["task"] == "locomotion":
            source["motion_file"] = str(
                amass_motion_file or data / "amassx/amass_smplx_train.pt"
            )
            source.pop("motion_file_shard_indices", None)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--hoi-teacher-dir", type=Path, required=True)
    parser.add_argument("--loco-teacher-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--amass-motion-file",
        type=Path,
        help="Full AMASS package; defaults to DATA_DIR/amassx/amass_smplx_train.pt",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Skip checking the AMASS file exists",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    template = yaml.safe_load(
        (root / "examples/experiments/gcc/s1_joint_pvq_sources_euler.yaml").read_text()
    )
    data = args.data_dir.resolve()
    amass = (
        args.amass_motion_file.resolve()
        if args.amass_motion_file is not None
        else data / "amassx/amass_smplx_train.pt"
    )
    if not args.manifest_only and not amass.is_file():
        raise FileNotFoundError(
            f"Upload the original full AMASS package first: {amass}"
        )
    if sum(s["task"] == "locomotion" for s in template["sources"]) != 1:
        raise ValueError("Expected exactly one locomotion source")
    manifest = relocate_manifest(
        template,
        data,
        args.hoi_teacher_dir.resolve(),
        args.loco_teacher_dir.resolve(),
        amass,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    content = yaml.safe_dump(manifest, sort_keys=False)
    if args.output.exists() and args.output.read_text() != content:
        raise ValueError(f"Refusing to replace a different manifest: {args.output}")
    args.output.write_text(content)
    print(f"Wrote {len(manifest['sources'])} sources to {args.output}")


if __name__ == "__main__":
    main()
