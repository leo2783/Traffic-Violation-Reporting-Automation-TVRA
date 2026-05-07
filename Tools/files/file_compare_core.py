"""Generic file comparison, copy, and delete engine.

This module is intentionally domain-agnostic.  It does not assume YOLO labels,
OCR outputs, images, or any specific dataset structure.  Files are compared by
configurable keys and then actions can be planned/executed with dry-run safety.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable


logger = logging.getLogger(__name__)


class CompareMode(str, Enum):
    """Supported comparison key strategies."""

    RELATIVE_PATH = "relative-path"
    FULL_NAME = "full-name"
    BASE_NAME = "base-name"
    CONTENT_HASH = "content-hash"


class Operation(str, Enum):
    """Supported generic file operations."""

    REPORT_ONLY = "report-only"
    COPY_SOURCE_ONLY = "copy-source-only"
    COPY_TARGET_ONLY = "copy-target-only"
    COPY_MATCHED_SOURCE = "copy-matched-source"
    DELETE_SOURCE_ONLY = "delete-source-only"
    DELETE_TARGET_ONLY = "delete-target-only"
    DELETE_TARGET_MATCHES = "delete-target-matches"


@dataclass(frozen=True)
class FileRecord:
    root: str
    path: str
    relative_path: str
    key: str
    size: int


@dataclass(frozen=True)
class PlannedAction:
    operation: str
    source: str
    destination: str | None = None
    reason: str = ""


@dataclass
class FileCompareResult:
    compare_mode: str
    operation: str
    dry_run: bool
    source_directories: list[str]
    target_directory: str
    destination_directory: str | None
    source_file_count: int
    target_file_count: int
    matched_source_files: list[str] = field(default_factory=list)
    matched_target_files: list[str] = field(default_factory=list)
    source_only_files: list[str] = field(default_factory=list)
    target_only_files: list[str] = field(default_factory=list)
    planned_actions: list[PlannedAction] = field(default_factory=list)
    executed_actions: list[PlannedAction] = field(default_factory=list)
    accelerator_used: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ContentHasher:
    """Content hashing with optional C++ acceleration and matching Python fallback."""

    def __init__(self, accelerator_path: Path | None = None) -> None:
        self.accelerator_path = accelerator_path
        self.accelerator_used = False

    def hash_many(self, files: Iterable[Path]) -> dict[Path, str]:
        file_list = list(files)
        if self.accelerator_path and self.accelerator_path.is_file():
            try:
                result = self._hash_many_cpp(file_list)
                self.accelerator_used = True
                return result
            except Exception as exc:  # noqa: BLE001 - must be safe fallback.
                logger.warning("C++ accelerator failed; fallback to Python FNV-1a: %s", exc)
        self.accelerator_used = False
        return {path: self._fnv1a_64(path) for path in file_list}

    def _hash_many_cpp(self, files: list[Path]) -> dict[Path, str]:
        completed = subprocess.run(  # noqa: S603 - local user-selected accelerator.
            [str(self.accelerator_path), *[str(path) for path in files]],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        hashes: dict[Path, str] = {}
        for line in completed.stdout.splitlines():
            if not line.strip():
                continue
            digest, raw_path = line.split("\t", maxsplit=1)
            hashes[Path(raw_path)] = digest
        if len(hashes) != len(files):
            raise RuntimeError("accelerator returned an incomplete hash list")
        return hashes

    @staticmethod
    def _fnv1a_64(path: Path) -> str:
        hash_value = 14_695_981_039_346_656_037
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                for byte in chunk:
                    hash_value ^= byte
                    hash_value = (hash_value * 1_099_511_628_211) & 0xFFFFFFFFFFFFFFFF
        return f"{hash_value:016x}"


class GenericFileCompareEngine:
    """Plan and execute generic copy/delete actions based on file comparison."""

    def __init__(self, accelerator_path: Path | None = None) -> None:
        self._hasher = ContentHasher(accelerator_path)

    def run(
        self,
        source_directories: list[Path],
        target_directory: Path,
        compare_mode: CompareMode,
        operation: Operation = Operation.REPORT_ONLY,
        destination_directory: Path | None = None,
        dry_run: bool = True,
        recursive: bool = True,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        report_path: Path | None = None,
    ) -> FileCompareResult:
        self._validate(source_directories, target_directory, operation, destination_directory)

        sources = self._scan_many(source_directories, compare_mode, recursive, include_patterns, exclude_patterns)
        targets = self._scan_one(target_directory, compare_mode, recursive, include_patterns, exclude_patterns)
        source_by_key = self._group_by_key(sources)
        target_by_key = self._group_by_key(targets)
        source_keys = set(source_by_key)
        target_keys = set(target_by_key)

        matched_source = [record for key in source_keys & target_keys for record in source_by_key[key]]
        matched_target = [record for key in source_keys & target_keys for record in target_by_key[key]]
        source_only = [record for key in source_keys - target_keys for record in source_by_key[key]]
        target_only = [record for key in target_keys - source_keys for record in target_by_key[key]]
        planned = self._plan_actions(
            operation=operation,
            matched_source=matched_source,
            matched_target=matched_target,
            source_only=source_only,
            target_only=target_only,
            destination_directory=destination_directory,
        )
        executed = self._execute_actions(planned, dry_run)

        result = FileCompareResult(
            compare_mode=compare_mode.value,
            operation=operation.value,
            dry_run=dry_run,
            source_directories=[str(path) for path in source_directories],
            target_directory=str(target_directory),
            destination_directory=str(destination_directory) if destination_directory else None,
            source_file_count=len(sources),
            target_file_count=len(targets),
            matched_source_files=[record.path for record in matched_source],
            matched_target_files=[record.path for record in matched_target],
            source_only_files=[record.path for record in source_only],
            target_only_files=[record.path for record in target_only],
            planned_actions=planned,
            executed_actions=executed,
            accelerator_used=self._hasher.accelerator_used,
        )
        if report_path:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    def _scan_many(
        self,
        directories: list[Path],
        compare_mode: CompareMode,
        recursive: bool,
        include_patterns: list[str] | None,
        exclude_patterns: list[str] | None,
    ) -> list[FileRecord]:
        records: list[FileRecord] = []
        for directory in directories:
            records.extend(self._scan_one(directory, compare_mode, recursive, include_patterns, exclude_patterns))
        return records

    def _scan_one(
        self,
        directory: Path,
        compare_mode: CompareMode,
        recursive: bool,
        include_patterns: list[str] | None,
        exclude_patterns: list[str] | None,
    ) -> list[FileRecord]:
        pattern = "**/*" if recursive else "*"
        files = sorted(path for path in directory.glob(pattern) if path.is_file())
        files = [path for path in files if self._is_included(path, include_patterns, exclude_patterns)]
        hashes = self._hasher.hash_many(files) if compare_mode == CompareMode.CONTENT_HASH else {}
        records: list[FileRecord] = []
        for path in files:
            relative_path = path.relative_to(directory).as_posix()
            key = self._make_key(path, relative_path, compare_mode, hashes)
            records.append(
                FileRecord(
                    root=str(directory),
                    path=str(path),
                    relative_path=relative_path,
                    key=key,
                    size=path.stat().st_size,
                )
            )
        return records

    @staticmethod
    def _make_key(
        path: Path,
        relative_path: str,
        compare_mode: CompareMode,
        hashes: dict[Path, str],
    ) -> str:
        if compare_mode == CompareMode.RELATIVE_PATH:
            return relative_path.lower()
        if compare_mode == CompareMode.FULL_NAME:
            return path.name.lower()
        if compare_mode == CompareMode.BASE_NAME:
            return path.stem.lower()
        return hashes[path]

    @staticmethod
    def _is_included(path: Path, include_patterns: list[str] | None, exclude_patterns: list[str] | None) -> bool:
        if include_patterns and not any(path.match(pattern) for pattern in include_patterns):
            return False
        if exclude_patterns and any(path.match(pattern) for pattern in exclude_patterns):
            return False
        return True

    @staticmethod
    def _group_by_key(records: list[FileRecord]) -> dict[str, list[FileRecord]]:
        grouped: dict[str, list[FileRecord]] = {}
        for record in records:
            grouped.setdefault(record.key, []).append(record)
        return grouped

    @staticmethod
    def _plan_actions(
        operation: Operation,
        matched_source: list[FileRecord],
        matched_target: list[FileRecord],
        source_only: list[FileRecord],
        target_only: list[FileRecord],
        destination_directory: Path | None,
    ) -> list[PlannedAction]:
        if operation == Operation.REPORT_ONLY:
            return []
        if operation == Operation.COPY_SOURCE_ONLY:
            return GenericFileCompareEngine._copy_actions(source_only, destination_directory, "source-only")
        if operation == Operation.COPY_TARGET_ONLY:
            return GenericFileCompareEngine._copy_actions(target_only, destination_directory, "target-only")
        if operation == Operation.COPY_MATCHED_SOURCE:
            return GenericFileCompareEngine._copy_actions(matched_source, destination_directory, "matched-source")
        if operation == Operation.DELETE_SOURCE_ONLY:
            return [PlannedAction(operation="delete", source=record.path, reason="source-only") for record in source_only]
        if operation == Operation.DELETE_TARGET_ONLY:
            return [PlannedAction(operation="delete", source=record.path, reason="target-only") for record in target_only]
        if operation == Operation.DELETE_TARGET_MATCHES:
            return [PlannedAction(operation="delete", source=record.path, reason="target-match") for record in matched_target]
        raise ValueError(f"Unsupported operation: {operation}")

    @staticmethod
    def _copy_actions(records: list[FileRecord], destination_directory: Path | None, reason: str) -> list[PlannedAction]:
        if destination_directory is None:
            raise ValueError("copy operations require destination_directory")
        return [
            PlannedAction(
                operation="copy",
                source=record.path,
                destination=str(destination_directory / record.relative_path),
                reason=reason,
            )
            for record in records
        ]

    @staticmethod
    def _execute_actions(actions: list[PlannedAction], dry_run: bool) -> list[PlannedAction]:
        if dry_run:
            return []
        executed: list[PlannedAction] = []
        for action in actions:
            source = Path(action.source)
            if action.operation == "copy":
                if action.destination is None:
                    raise ValueError("copy action missing destination")
                destination = Path(action.destination)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                executed.append(action)
            elif action.operation == "delete":
                if source.exists() and source.is_file():
                    source.unlink()
                    executed.append(action)
            else:
                raise ValueError(f"Unsupported planned action: {action.operation}")
        return executed

    @staticmethod
    def _validate(
        source_directories: list[Path],
        target_directory: Path,
        operation: Operation,
        destination_directory: Path | None,
    ) -> None:
        if not source_directories:
            raise ValueError("at least one source directory is required")
        for directory in source_directories:
            if not directory.is_dir():
                raise NotADirectoryError(f"source directory not found: {directory}")
        if not target_directory.is_dir():
            raise NotADirectoryError(f"target directory not found: {target_directory}")
        if operation in {Operation.COPY_SOURCE_ONLY, Operation.COPY_TARGET_ONLY, Operation.COPY_MATCHED_SOURCE} and destination_directory is None:
            raise ValueError("copy operation requires destination directory")


def parse_patterns(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return [part.strip() for part in raw.split(";") if part.strip()]


def summarize_result(result: FileCompareResult) -> str:
    return "\n".join(
        [
            f"compare_mode: {result.compare_mode}",
            f"operation: {result.operation}",
            f"dry_run: {result.dry_run}",
            f"source_file_count: {result.source_file_count}",
            f"target_file_count: {result.target_file_count}",
            f"matched_source_files: {len(result.matched_source_files)}",
            f"matched_target_files: {len(result.matched_target_files)}",
            f"source_only_files: {len(result.source_only_files)}",
            f"target_only_files: {len(result.target_only_files)}",
            f"planned_actions: {len(result.planned_actions)}",
            f"executed_actions: {len(result.executed_actions)}",
            f"accelerator_used: {result.accelerator_used}",
        ]
    )