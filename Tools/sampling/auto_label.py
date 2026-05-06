"""Auto Label workflow models, selection logic, and annotation writers."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

try:
    from .utils import FeatureExtractor, YoloAnalyzer, path_check
except ImportError:
    from utils import FeatureExtractor, YoloAnalyzer, path_check


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectionBox:
    """Normalized YOLO-style detection box plus pixel coordinates."""

    class_id: int
    class_name: str
    confidence: float
    x_center: float
    y_center: float
    width: float
    height: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass(frozen=True)
class AutoLabelCandidate:
    """A high-confidence image candidate selected for auto-label output."""

    image_path: Path
    image_name: str
    image_width: int
    image_height: int
    max_confidence: float
    boxes: list[DetectionBox]


class AutoLabelSelector:
    """Select diverse auto-label candidates using embedding similarity deduplication."""

    def __init__(self, similarity_threshold: float = 0.9) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold 必須介於 0.0 到 1.0 之間")
        self._similarity_threshold = similarity_threshold
        self._feature_extractor = FeatureExtractor()

    def select(
        self,
        candidates: list[AutoLabelCandidate],
        progress_callback: Any | None = None,
    ) -> list[AutoLabelCandidate]:
        """Return candidates after removing highly similar images."""

        if not candidates:
            return []

        embeddings: list[np.ndarray] = []
        valid_candidates: list[AutoLabelCandidate] = []
        total = len(candidates)

        logger.info("開始抽取 Auto Label 候選圖片特徵...")
        for index, candidate in enumerate(candidates):
            if path_check(str(candidate.image_path)):
                embedding = self._feature_extractor.extract_embedding(str(candidate.image_path))
                if embedding is not None:
                    embeddings.append(embedding)
                    valid_candidates.append(candidate)
            if progress_callback:
                progress_callback(index + 1, total)

        if not embeddings:
            logger.warning("Auto Label 沒有有效 embedding，無法選樣。")
            return []

        embeddings_matrix = torch.tensor(
            np.array(embeddings),
            device=self._feature_extractor.device,
        )
        keep_indices = self._calculate_keep_indices(embeddings_matrix)
        selected = [valid_candidates[index] for index in keep_indices]
        logger.info(
            "Auto Label similarity 去重完成：候選 %s 張，保留 %s 張。",
            len(valid_candidates),
            len(selected),
        )
        return selected

    def _calculate_keep_indices(self, embeddings_matrix: torch.Tensor) -> np.ndarray:
        norms = torch.norm(embeddings_matrix, dim=1, keepdim=True) + 1e-10
        normalized = embeddings_matrix / norms
        sim_matrix = torch.mm(normalized, normalized.t())
        upper_tri = torch.triu(sim_matrix, diagonal=1)
        duplicates = torch.any(upper_tri > self._similarity_threshold, dim=0)
        return np.where(~duplicates.cpu().numpy())[0]


class ImageCopyWriter:
    """Copy selected images while preserving original file extensions."""

    def write(self, candidate: AutoLabelCandidate, output_root: Path) -> Path:
        images_dir = output_root / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        destination = images_dir / candidate.image_path.name
        shutil.copy2(candidate.image_path, destination)
        return destination


class YoloTxtWriter:
    """Write YOLO detection labels for selected candidates."""

    def __init__(self, keep_confidence: bool = False) -> None:
        self._keep_confidence = keep_confidence

    def write(self, candidate: AutoLabelCandidate, output_root: Path) -> Path:
        labels_dir = output_root / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        destination = labels_dir / f"{candidate.image_path.stem}.txt"
        lines = []
        for box in candidate.boxes:
            values = [
                str(box.class_id),
                f"{box.x_center:.6f}",
                f"{box.y_center:.6f}",
                f"{box.width:.6f}",
                f"{box.height:.6f}",
            ]
            if self._keep_confidence:
                values.append(f"{box.confidence:.6f}")
            lines.append(" ".join(values))
        destination.write_text("\n".join(lines), encoding="utf-8")
        return destination


class AnyLabelJsonWriter:
    """Write LabelMe/AnyLabel-style JSON annotation files."""

    def __init__(self, version: str = "0.4.36") -> None:
        self._version = version

    def write(self, candidate: AutoLabelCandidate, output_root: Path) -> Path:
        json_dir = output_root / "anylabel_json"
        json_dir.mkdir(parents=True, exist_ok=True)
        destination = json_dir / f"{candidate.image_path.stem}.json"
        payload = {
            "version": self._version,
            "flags": {},
            "shapes": [self._shape_from_box(box) for box in candidate.boxes],
            "imagePath": candidate.image_path.name,
            "imageData": None,
            "imageHeight": candidate.image_height,
            "imageWidth": candidate.image_width,
        }
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return destination

    @staticmethod
    def _shape_from_box(box: DetectionBox) -> dict[str, Any]:
        return {
            "label": box.class_name or str(box.class_id),
            "text": "",
            "points": [
                [round(box.x_min, 3), round(box.y_min, 3)],
                [round(box.x_max, 3), round(box.y_min, 3)],
                [round(box.x_max, 3), round(box.y_max, 3)],
                [round(box.x_min, 3), round(box.y_max, 3)],
            ],
            "group_id": None,
            "shape_type": "polygon",
            "flags": {"confidence": round(box.confidence, 6)},
        }


class AutoLabelWorkflow:
    """Build, select, and write automatic labels from YOLO detections."""

    def __init__(
        self,
        yolo_weights: str,
        confidence_threshold: float = 0.8,
        similarity_threshold: float = 0.9,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold 必須介於 0.0 到 1.0 之間")
        self._confidence_threshold = confidence_threshold
        self._analyzer = YoloAnalyzer(yolo_weights)
        self._selector = AutoLabelSelector(similarity_threshold=similarity_threshold)

    def run(
        self,
        image_paths: list[str],
        output_root: Path,
        copy_images: bool = True,
        output_yolo_txt: bool = False,
        output_anylabel_json: bool = False,
        keep_confidence: bool = False,
        progress_callback: Any | None = None,
    ) -> list[AutoLabelCandidate]:
        """Execute Auto Label workflow and write requested outputs."""

        output_root.mkdir(parents=True, exist_ok=True)
        candidates = self.build_candidates(image_paths)
        selected = self._selector.select(candidates, progress_callback=progress_callback)
        self.write_outputs(
            selected,
            output_root=output_root,
            copy_images=copy_images,
            output_yolo_txt=output_yolo_txt,
            output_anylabel_json=output_anylabel_json,
            keep_confidence=keep_confidence,
        )
        self.write_report(
            output_root=output_root,
            total_images=len(image_paths),
            candidates=candidates,
            selected=selected,
            copy_images=copy_images,
            output_yolo_txt=output_yolo_txt,
            output_anylabel_json=output_anylabel_json,
            keep_confidence=keep_confidence,
        )
        return selected

    def build_candidates(self, image_paths: list[str]) -> list[AutoLabelCandidate]:
        """Run YOLO inference and build high-confidence candidates."""

        if not image_paths:
            return []

        logger.info("開始建立 Auto Label 候選，共 %s 張圖片...", len(image_paths))
        candidates: list[AutoLabelCandidate] = []
        results = self._analyzer.predict(image_paths, verbose=False)
        names = getattr(self._analyzer._model, "names", {})  # noqa: SLF001

        for result in results:
            image_path = Path(result.path)
            boxes = self._boxes_from_result(result, names)
            boxes = [box for box in boxes if box.confidence >= self._confidence_threshold]
            if not boxes:
                continue

            width, height = self._image_size(image_path, result)
            candidates.append(
                AutoLabelCandidate(
                    image_path=image_path,
                    image_name=image_path.stem,
                    image_width=width,
                    image_height=height,
                    max_confidence=max(box.confidence for box in boxes),
                    boxes=boxes,
                )
            )

        logger.info("Auto Label 高信心候選數量: %s", len(candidates))
        return candidates

    def write_outputs(
        self,
        candidates: list[AutoLabelCandidate],
        output_root: Path,
        copy_images: bool,
        output_yolo_txt: bool,
        output_anylabel_json: bool,
        keep_confidence: bool,
    ) -> None:
        writers = []
        if copy_images:
            writers.append(ImageCopyWriter())
        if output_yolo_txt:
            writers.append(YoloTxtWriter(keep_confidence=keep_confidence))
        if output_anylabel_json:
            writers.append(AnyLabelJsonWriter())

        for candidate in candidates:
            for writer in writers:
                writer.write(candidate, output_root)

    @staticmethod
    def write_report(
        output_root: Path,
        total_images: int,
        candidates: list[AutoLabelCandidate],
        selected: list[AutoLabelCandidate],
        copy_images: bool,
        output_yolo_txt: bool,
        output_anylabel_json: bool,
        keep_confidence: bool,
    ) -> Path:
        report_path = output_root / "auto_label_report.json"
        payload = {
            "total_images": total_images,
            "high_confidence_candidates": len(candidates),
            "selected_candidates": len(selected),
            "copy_images": copy_images,
            "output_yolo_txt": output_yolo_txt,
            "output_anylabel_json": output_anylabel_json,
            "keep_confidence": keep_confidence,
            "selected": [
                {
                    "image_path": str(candidate.image_path),
                    "image_name": candidate.image_name,
                    "image_width": candidate.image_width,
                    "image_height": candidate.image_height,
                    "max_confidence": candidate.max_confidence,
                    "box_count": len(candidate.boxes),
                }
                for candidate in selected
            ],
        }
        report_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return report_path

    @staticmethod
    def _boxes_from_result(result: Any, names: dict[int, str]) -> list[DetectionBox]:
        image_height, image_width = result.orig_shape
        detections: list[DetectionBox] = []
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        xywhn = result.boxes.xywhn.cpu().numpy()
        xyxy = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for index, class_id in enumerate(classes):
            x_center, y_center, width, height = xywhn[index]
            x_min, y_min, x_max, y_max = xyxy[index]
            detections.append(
                DetectionBox(
                    class_id=int(class_id),
                    class_name=str(names.get(int(class_id), class_id)),
                    confidence=float(confidences[index]),
                    x_center=float(x_center),
                    y_center=float(y_center),
                    width=float(width),
                    height=float(height),
                    x_min=float(x_min),
                    y_min=float(y_min),
                    x_max=float(x_max),
                    y_max=float(y_max),
                )
            )
        return detections

    @staticmethod
    def _image_size(image_path: Path, result: Any) -> tuple[int, int]:
        if result.orig_shape:
            height, width = result.orig_shape
            return int(width), int(height)
        with Image.open(image_path) as image:
            return image.size


# Backward-compatible alias for old imports.
AutoLabelClassifier = AutoLabelSelector
