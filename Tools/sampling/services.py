"""Service layer shared by CLI entrypoints and the PyQt GUI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

try:  # package execution: python -m Tools.sampling.gui
    from .constants import collect_image_files, supported_image_extensions_text
except ImportError:  # script execution from Tools/sampling
    from constants import collect_image_files, supported_image_extensions_text


ProgressCallback = Callable[[int, int], None]

logger = logging.getLogger(__name__)


class DeduplicationService:
    """Image deduplication workflow service."""

    def __init__(self, threshold: float, yolo_weights: str | None = None):
        try:
            from .embedding import ImageDeduplicator
        except ImportError:
            from embedding import ImageDeduplicator

        self._deduplicator = ImageDeduplicator(
            threshold=threshold,
            yolo_weights=yolo_weights,
        )

    def execute(
        self,
        input_folder: Path,
        output_folder: Path,
        use_confidence: bool,
        sample_way: str = "negative",
        write_mode: str = "per-folder",
        progress_callback: Optional[ProgressCallback] = None,
    ) -> list[str]:
        """Run image deduplication and return kept image paths."""

        if not input_folder.exists() or not input_folder.is_dir():
            raise NotADirectoryError(f"輸入資料夾不存在或無效: {input_folder}")

        file_list = collect_image_files(input_folder)
        if not file_list:
            logger.warning(
                "輸入資料夾內沒有有效的圖片檔案 (支援格式: %s)",
                supported_image_extensions_text(),
            )
            return []

        logger.info("開始執行圖片去重分析...")
        output_folder.mkdir(parents=True, exist_ok=True)

        final_files = self._deduplicator.process_batch(
            file_list,
            use_confidence=use_confidence,
            sample_way=sample_way,
            write_mode=write_mode,
            output_folder=str(output_folder),
            progress_callback=progress_callback,
        )

        logger.info("原始圖片數量: %s", len(file_list))
        logger.info("去重後總計保留數量: %s", len(final_files))
        logger.info("-" * 30)
        logger.info("所有乾淨圖片已處理完畢！")
        return final_files


class NegativeSamplingService:
    """Negative sample extraction workflow service."""

    def __init__(self, yolo_weights: str, temperature: float = 5.0):
        try:
            from .extract_negative import NegativeSampler
        except ImportError:
            from extract_negative import NegativeSampler

        self._sampler = NegativeSampler(
            yolo_weights=yolo_weights,
            temperature=temperature,
        )

    def execute(
        self,
        input_folder: Path,
        output_folder: Path,
        num_samples: int,
    ) -> list[str]:
        """Run negative sampling and return copied output paths."""

        if not input_folder.exists() or not input_folder.is_dir():
            raise NotADirectoryError(f"輸入資料夾不存在或無效: {input_folder}")
        if num_samples <= 0:
            raise ValueError("抽樣數量必須大於 0")

        image_paths = collect_image_files(input_folder)
        if not image_paths:
            logger.warning(
                "輸入資料夾內沒有有效的圖片檔案 (支援格式: %s)",
                supported_image_extensions_text(),
            )
            return []

        output_folder.mkdir(parents=True, exist_ok=True)
        return self._sampler.sample(
            image_paths=image_paths,
            num_samples=num_samples,
            output_dir=str(output_folder),
        )


class ValidationCleanService:
    """YOLO validation-set cleaning workflow service."""

    def __init__(self, yolo_weights: str, threshold: float = 0.6):
        try:
            from .val_clean import ValidationCleaner
        except ImportError:
            from val_clean import ValidationCleaner

        self._cleaner = ValidationCleaner(
            yolo_weights=yolo_weights,
            threshold=threshold,
        )

    def execute(self, source_path: Path, output_path: Path) -> None:
        """Run validation cleaning."""

        self._cleaner.clean(
            source_path=str(source_path),
            out_path=str(output_path),
        )


class YoloTestService:
    """YOLO inference test workflow service."""

    def __init__(self, yolo_weights: str, conf: float = 0.7):
        try:
            from .test_runner import UnifiedTestRunner
        except ImportError:
            from test_runner import UnifiedTestRunner

        self._runner = UnifiedTestRunner(yolo_weights=yolo_weights, conf=conf)

    def execute(
        self,
        source_type: str,
        path: Path | None = None,
        count: int = 5,
        output: Path | None = None,
    ) -> None:
        """Run the requested YOLO test source."""

        output_text = str(output) if output else None
        if source_type == "video":
            if path is None:
                raise ValueError("source_type=video 需要指定影片資料夾")
            self._runner.test_video_folder(str(path), output_text)
        elif source_type == "image":
            if path is None:
                raise ValueError("source_type=image 需要指定圖片資料夾")
            self._runner.test_image_folder(str(path), output_text)
        elif source_type == "file":
            if path is None:
                raise ValueError("source_type=file 需要指定單一檔案")
            self._runner.test_single_file(str(path), output_text)
        elif source_type == "youtube":
            self._runner.test_youtube(count=count, output=output_text)
        else:
            raise ValueError(f"未知的 YOLO 測試來源類型: {source_type}")


class AutoLabelService:
    """Auto Label workflow service for GUI and future CLI integration."""

    def __init__(
        self,
        yolo_weights: str,
        confidence_threshold: float = 0.8,
        similarity_threshold: float = 0.9,
    ) -> None:
        try:
            from .auto_label import AutoLabelWorkflow
        except ImportError:
            from auto_label import AutoLabelWorkflow

        self._workflow = AutoLabelWorkflow(
            yolo_weights=yolo_weights,
            confidence_threshold=confidence_threshold,
            similarity_threshold=similarity_threshold,
        )

    def execute(
        self,
        input_folder: Path,
        output_folder: Path,
        copy_images: bool = True,
        output_yolo_txt: bool = False,
        output_anylabel_json: bool = False,
        keep_confidence: bool = False,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> list[object]:
        """Run Auto Label on supported images in a folder."""

        if not input_folder.exists() or not input_folder.is_dir():
            raise NotADirectoryError(f"輸入資料夾不存在或無效: {input_folder}")

        image_paths = collect_image_files(input_folder)
        if not image_paths:
            logger.warning(
                "輸入資料夾內沒有有效的圖片檔案 (支援格式: %s)",
                supported_image_extensions_text(),
            )
            return []

        if not (copy_images or output_yolo_txt or output_anylabel_json):
            raise ValueError("至少需要選擇一種輸出：圖片、YOLO txt 或 AnyLabel json")

        return self._workflow.run(
            image_paths=image_paths,
            output_root=output_folder,
            copy_images=copy_images,
            output_yolo_txt=output_yolo_txt,
            output_anylabel_json=output_anylabel_json,
            keep_confidence=keep_confidence,
            progress_callback=progress_callback,
        )