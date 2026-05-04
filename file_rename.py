import os
import argparse
import uuid
from pathlib import Path

def rename_paired_files(dir1_path, dir2_path, start_idx=0):
    dir1 = Path(dir1_path)
    dir2 = Path(dir2_path)

    if not dir1.exists() or not dir1.is_dir():
        print(f"Directory not found or is not a directory: {dir1}")
        return
    if not dir2.exists() or not dir2.is_dir():
        print(f"Directory not found or is not a directory: {dir2}")
        return

    # Find all files in both directories
    files1 = {f.stem: f for f in dir1.iterdir() if f.is_file()}
    files2 = {f.stem: f for f in dir2.iterdir() if f.is_file()}

    # Find common stems (base names without extensions)
    common_stems = sorted(list(set(files1.keys()).intersection(set(files2.keys()))))

    if not common_stems:
        print(f"No matching file pairs found between '{dir1}' and '{dir2}'.")
        return

    print(f"Found {len(common_stems)} paired files to rename.")

    # Phase 1: Rename all paired files to a temporary UUID name
    # This prevents any accidental overwriting if files with target names (like 00000.jpg) already exist.
    temp_pairs = []
    for stem in common_stems:
        f1 = files1[stem]
        f2 = files2[stem]

        temp_stem = str(uuid.uuid4())
        temp_f1 = dir1 / f"{temp_stem}{f1.suffix}"
        temp_f2 = dir2 / f"{temp_stem}{f2.suffix}"

        # Rename to temp
        f1.rename(temp_f1)
        f2.rename(temp_f2)

        temp_pairs.append((temp_f1, temp_f2))

    # Phase 2: Rename from temporary UUID to final sequential names
    for idx, (temp_f1, temp_f2) in enumerate(temp_pairs, start=start_idx):
        new_stem = f"{idx:05d}"
        
        final_f1 = dir1 / f"{new_stem}{temp_f1.suffix}"
        final_f2 = dir2 / f"{new_stem}{temp_f2.suffix}"

        temp_f1.rename(final_f1)
        temp_f2.rename(final_f2)

    print(f"Successfully renamed {len(common_stems)} file pairs. (From {start_idx:05d} to {start_idx + len(common_stems) - 1:05d})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename paired files in two directories sequentially.")
    parser.add_argument("dir1", type=str, help="Path to the first directory (e.g., images)")
    parser.add_argument("dir2", type=str, help="Path to the second directory (e.g., labels)")
    parser.add_argument("--start", type=int, default=0, help="Starting index for renaming (default: 0)")

    args = parser.parse_args()

    rename_paired_files(args.dir1, args.dir2, start_idx=args.start)
