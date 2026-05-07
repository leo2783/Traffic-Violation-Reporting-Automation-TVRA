"""
Extract images listed in a markdown problem report.

用途：
    從 task.md 這類 Markdown 回報中解析「檔案: 00001.txt」或「00001.txt | ...」清單，
    再到使用者指定的圖片根資料夾底下遞迴尋找同名圖片，並複製到輸出資料夾。

注意：
    - 本工具只複製圖片，不清洗、不修改、不刪除原始資料。
    - 使用者通常只需要修改下方「使用者設定區」的 IMAGE_ROOT。
"""

from __future__ import annotations

import re
import shutil
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ===== 使用者設定區：通常只需要改這裡 =====
TASK_MARKDOWN_PATH = PROJECT_ROOT / "task.md"
IMAGE_ROOT = Path(r"C:/Users/qet63/Pictures/poly")
OUTPUT_DIR = PROJECT_ROOT / "extracted_invalid_polygon_images"
# ======================================

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

LABEL_PATTERN = re.compile(r"(?:檔案:\s*)?([^\s|]+\.txt)", re.IGNORECASE)


def parse_problem_label_stems(markdown_path: Path) -> list[str]:
    """Parse markdown report and return unique label stems in original order."""
    if not markdown_path.is_file():
        raise FileNotFoundError(f"找不到 Markdown 回報檔案：{markdown_path}")

    stems: list[str] = []
    seen: set[str] = set()

    for line in markdown_path.read_text(encoding="utf-8").splitlines():
        match = LABEL_PATTERN.search(line)
        if not match:
            continue

        stem = Path(match.group(1)).stem
        if stem not in seen:
            stems.append(stem)
            seen.add(stem)

    return stems


def build_image_index(image_root: Path) -> dict[str, list[Path]]:
    """Recursively index all supported images by filename stem."""
    if not image_root.is_dir():
        raise NotADirectoryError(f"找不到圖片根資料夾：{image_root}")

    image_index: dict[str, list[Path]] = defaultdict(list)
    for image_path in image_root.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            image_index[image_path.stem].append(image_path)

    return dict(image_index)


def unique_destination(output_dir: Path, image_root: Path, image_path: Path, duplicated: bool) -> Path:
    """Return a destination path and avoid overwriting when duplicated stems exist."""
    if not duplicated:
        return output_dir / image_path.name

    relative_path = image_path.relative_to(image_root)
    return output_dir / relative_path


def extract_images(label_stems: list[str], image_index: dict[str, list[Path]], image_root: Path, output_dir: Path) -> tuple[int, list[str], dict[str, list[Path]]]:
    """Copy matched images to output_dir.

    Returns:
        copied_count: copied image count
        missing_stems: stems not found in image_index
        duplicated_matches: stems matching more than one image
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    missing_stems: list[str] = []
    duplicated_matches: dict[str, list[Path]] = {}

    for stem in label_stems:
        matched_paths = image_index.get(stem, [])
        if not matched_paths:
            missing_stems.append(stem)
            continue

        duplicated = len(matched_paths) > 1
        if duplicated:
            duplicated_matches[stem] = matched_paths

        for image_path in matched_paths:
            destination = unique_destination(output_dir, image_root, image_path, duplicated)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, destination)
            copied_count += 1

    return copied_count, missing_stems, duplicated_matches


def print_summary(total_labels: int, indexed_images: int, copied_count: int, missing_stems: list[str], duplicated_matches: dict[str, list[Path]]) -> None:
    """Print a concise execution summary."""
    print("\n===== 提取摘要 =====")
    print(f"Markdown 問題檔名數量：{total_labels}")
    print(f"圖片根資料夾索引圖片數量：{indexed_images}")
    print(f"成功複製圖片數量：{copied_count}")
    print(f"找不到對應圖片數量：{len(missing_stems)}")
    print(f"同名圖片重複 stem 數量：{len(duplicated_matches)}")

    if missing_stems:
        print("\n找不到對應圖片的 stem：")
        for stem in missing_stems:
            print(f"- {stem}")

    if duplicated_matches:
        print("\n注意：以下 stem 在圖片根資料夾中找到多張圖片，已保留相對資料夾結構避免覆蓋：")
        for stem, paths in duplicated_matches.items():
            print(f"- {stem}")
            for path in paths:
                print(f"  - {path}")


def main() -> None:
    """Run extraction with the paths configured in the user settings section."""
    print(f"讀取 Markdown 回報：{TASK_MARKDOWN_PATH}")
    label_stems = parse_problem_label_stems(TASK_MARKDOWN_PATH)
    if not label_stems:
        print("沒有從 Markdown 回報中解析到任何問題檔名。")
        return

    print(f"遞迴索引圖片根資料夾：{IMAGE_ROOT}")
    image_index = build_image_index(IMAGE_ROOT)

    print(f"複製圖片到：{OUTPUT_DIR}")
    copied_count, missing_stems, duplicated_matches = extract_images(
        label_stems=label_stems,
        image_index=image_index,
        image_root=IMAGE_ROOT,
        output_dir=OUTPUT_DIR,
    )

    indexed_images = sum(len(paths) for paths in image_index.values())
    print_summary(
        total_labels=len(label_stems),
        indexed_images=indexed_images,
        copied_count=copied_count,
        missing_stems=missing_stems,
        duplicated_matches=duplicated_matches,
    )


if __name__ == "__main__":
    main()