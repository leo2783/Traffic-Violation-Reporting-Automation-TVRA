"""Shared constants and file discovery helpers for the sampling module."""

from __future__ import annotations

from pathlib import Path


SUPPORTED_IMAGE_EXTENSIONS: set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

SUPPORTED_VIDEO_EXTENSIONS: set[str] = {
    ".mp4",
    ".avi",
    ".mov",
    ".ts",
}


def is_supported_image_file(path: str | Path) -> bool:
    """Return True when path exists and has a supported image extension."""

    candidate = Path(path)
    return candidate.is_file() and candidate.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def is_supported_video_file(path: str | Path) -> bool:
    """Return True when path exists and has a supported video extension."""

    candidate = Path(path)
    return candidate.is_file() and candidate.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS


def collect_image_files(folder: str | Path) -> list[str]:
    """Collect supported image files from a folder as POSIX-style absolute paths."""

    directory = Path(folder)
    if not directory.is_dir():
        return []
    return sorted(
        file.resolve().as_posix()
        for file in directory.iterdir()
        if is_supported_image_file(file)
    )


def collect_video_files(folder: str | Path) -> list[str]:
    """Collect supported video files from a folder as POSIX-style absolute paths."""

    directory = Path(folder)
    if not directory.is_dir():
        return []
    return sorted(
        file.resolve().as_posix()
        for file in directory.iterdir()
        if is_supported_video_file(file)
    )


def supported_image_extensions_text() -> str:
    """Return a user-facing extension list."""

    return ", ".join(sorted(ext.lstrip(".") for ext in SUPPORTED_IMAGE_EXTENSIONS))