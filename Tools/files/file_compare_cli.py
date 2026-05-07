"""CLI for the generic file comparison/copy/delete tool."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

try:
    from .file_compare_core import CompareMode, GenericFileCompareEngine, Operation, parse_patterns, summarize_result
except ImportError:
    from file_compare_core import CompareMode, GenericFileCompareEngine, Operation, parse_patterns, summarize_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generic file compare/copy/delete utility")
    parser.add_argument("--source", action="append", required=True, help="Source directory. Repeat for multiple sources.")
    parser.add_argument("--target", required=True, help="Target directory")
    parser.add_argument("--destination", help="Destination directory for copy operations")
    parser.add_argument("--mode", choices=[mode.value for mode in CompareMode], default=CompareMode.RELATIVE_PATH.value)
    parser.add_argument("--operation", choices=[operation.value for operation in Operation], default=Operation.REPORT_ONLY.value)
    parser.add_argument("--execute", action="store_true", help="Execute planned copy/delete actions. Default is dry-run.")
    parser.add_argument("--no-recursive", action="store_true", help="Only scan top-level files")
    parser.add_argument("--include", help="Semicolon-separated glob patterns, e.g. *.jpg;*.json")
    parser.add_argument("--exclude", help="Semicolon-separated glob patterns")
    parser.add_argument("--report", help="Optional JSON report path")
    parser.add_argument("--accelerator", help="Optional C++ hash accelerator executable for content-hash mode")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()
    engine = GenericFileCompareEngine(Path(args.accelerator) if args.accelerator else None)
    result = engine.run(
        source_directories=[Path(source) for source in args.source],
        target_directory=Path(args.target),
        compare_mode=CompareMode(args.mode),
        operation=Operation(args.operation),
        destination_directory=Path(args.destination) if args.destination else None,
        dry_run=not args.execute,
        recursive=not args.no_recursive,
        include_patterns=parse_patterns(args.include),
        exclude_patterns=parse_patterns(args.exclude),
        report_path=Path(args.report) if args.report else None,
    )
    print(summarize_result(result))


if __name__ == "__main__":
    main()