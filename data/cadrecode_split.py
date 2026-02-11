import argparse
import pickle
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split dataset into 3 groups using cadrecode_group_split.pkl"
    )
    parser.add_argument(
        "--src_dataset",
        type=str,
        required=True,
        help="Path to source dataset (cad-recode-v1.5)",
    )
    parser.add_argument(
        "--dst_dataset",
        type=str,
        required=True,
        help="Path to destination dataset (cad-recode-v1.5-3groups)",
    )
    parser.add_argument(
        "--group_split",
        type=str,
        required=True,
        help='Path to cadrecode_group_split.pkl',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    SRC_ROOT = Path(args.src_dataset)
    DST_ROOT = Path(args.dst_dataset)

    TRAIN_SRC = SRC_ROOT / "train"
    VAL_SRC = SRC_ROOT / "val"

    # ----------------------------
    # Load group split
    # ----------------------------
    with open(args.group_split, "rb") as f:
        group_split = pickle.load(f)

    group0 = set(group_split.get("0", []))
    group1 = set(group_split.get("1", []))

    # ----------------------------
    # Create directory structure
    # ----------------------------
    for group_id in ["0", "1", "2"]:
        (DST_ROOT / group_id / "train").mkdir(parents=True, exist_ok=True)
        (DST_ROOT / group_id / "val").mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Copy val to each group
    # ----------------------------
    for group_id in ["0", "1", "2"]:
        dst_val = DST_ROOT / group_id / "val"

        for file in VAL_SRC.glob("*"):
            if file.suffix in [".py", ".stl"]:
                shutil.copy(file, dst_val / file.name)

    # ----------------------------
    # Process train directory
    # ----------------------------
    for batch_dir in TRAIN_SRC.iterdir():
        if not batch_dir.is_dir():
            continue

        for file in batch_dir.glob("*"):
            if file.suffix not in [".py", ".stl"]:
                continue

            stem = file.stem

            if stem in group0:
                group_id = "0"
            elif stem in group1:
                group_id = "1"
            else:
                group_id = "2"

            dst_train = DST_ROOT / group_id / "train"
            shutil.copy(file, dst_train / file.name)


if __name__ == "__main__":
    main()
