"""Build dataset index files for training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Allow running as a script without installing the package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.io import save_index

def build_index(mask_dir: Path, occ_dir: Path, cameras_json: Path, split: str) -> dict:
    with open(cameras_json, "r", encoding="utf-8") as f:
        cameras = json.load(f)
    items = []
    for entry in cameras[split]:
        mask_path = mask_dir / entry["mask_file"]
        occ_path = occ_dir / f"{entry['mesh_id']}.npz"
        items.append(
            {
                "mask_npy": str(mask_path),
                "K": entry["intrinsics"],
                "Rt": entry["extrinsics"],
                "occ_npz": str(occ_path),
                "cat": entry["category"],
                "cat_id": entry["category_id"],
            }
        )
    return {"items": items}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_dir", type=Path, required=True)
    parser.add_argument("--occ_dir", type=Path, required=True)
    parser.add_argument("--cams", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    index = build_index(args.mask_dir, args.occ_dir, args.cams, args.split)
    save_index(index, args.out)
