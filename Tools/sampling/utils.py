import os
import logging
from enum import Enum
from typing import List, Optional, Any, Callable
import numpy as np

import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image, ImageOps, UnidentifiedImageError

try:
    from .constants import is_supported_image_file
    from .interfaces import FeatureExtractorInterface, ObjectDetectorInterface
except ImportError:
    from constants import is_supported_image_file
    from interfaces import FeatureExtractorInterface, ObjectDetectorInterface

logger = logging.getLogger(__name__)

def get_device() -> torch.device:
    """取得運算設備"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def safe_image_open(image_path: str) -> Optional[Image.Image]:
    """安全地讀取圖片，若發生損毀則回傳 None"""
    try:
        img = Image.open(image_path).convert("RGB")
        return img
    except (UnidentifiedImageError, OSError) as e:
        logger.warning(f"無法讀取圖片 {image_path}: {e}")
        return None

class FeatureExtractor(FeatureExtractorInterface):
    """MobileNetV3 影像特徵擷取器"""
    def __init__(self) -> None:
        self._device = get_device()
        self._weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        self._model = models.mobilenet_v3_small(weights=self._weights)
        self._model.classifier = torch.nn.Identity()
        self._model.to(self._device)
        self._model.eval()
        
        self._preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @property
    def device(self) -> torch.device:
        return self._device

    def extract_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """讀取單張圖片，產出 1D 特徵向量 (回傳 numpy array)"""
        img = safe_image_open(image_path)
        if img is None:
            return None
            
        img = ImageOps.pad(img, (224, 224), color=(0, 0, 0), centering=(0.5, 0.5))
        input_tensor = self._preprocess(img).unsqueeze(0).to(self._device)
        
        with torch.no_grad():
            embedding = self._model(input_tensor)
            
        return embedding.squeeze().cpu().numpy()

    def extract_features_from_paths(self, image_paths: List[str], progress_callback: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
        """
        Batch processing: extract feature vectors for multiple images, return numpy matrix.
        批次處理：對多張圖片提取特徵向量，回傳 numpy 矩陣。
        """
        total = len(image_paths)
        features = []
        for idx, path in enumerate(image_paths):
            emb = self.extract_embedding(path)
            if emb is not None:
                features.append(emb)
            if progress_callback:
                progress_callback(idx + 1, total)
        if features:
            return np.vstack(features)
        return np.array([])
        
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(device={self._device})>"

class SampleStrategy(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"

class YoloAnalyzer(ObjectDetectorInterface):
    """YOLO 模型推論封裝 (支援批次 GPU 推論與 stream=True 避免 OOM)"""
    def __init__(self, model_path: str) -> None:
        if not isinstance(model_path, str):
            raise TypeError("model_path 必須是字串")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到 YOLO 模型權重檔案: {model_path}")
            
        from ultralytics import YOLO
        self._model = YOLO(model_path, task="detect")
        self._model_path = model_path
        
    def predict(self, source: Any, **kwargs) -> Any:
        """執行模型預測，強制使用 GPU 與 stream=True"""
        # 確保 kwargs 內有 device 設定
        if "device" not in kwargs:
            kwargs["device"] = "0" if torch.cuda.is_available() else "cpu"
        # 確保 stream=True 避免記憶體暴增
        if isinstance(source, list) and "stream" not in kwargs:
            kwargs["stream"] = True
        elif not isinstance(source, list) and "stream" not in kwargs:
            kwargs["stream"] = True
        return self._model.predict(source, **kwargs)

    def analyze(self, image_paths: List[str], sample_way: str, progress_callback: Optional[Callable[[int, int], None]] = None) -> List[tuple]:
        """
        批次分析影像列表並根據策略排序。
        使用批次推論 (stream=True) 而非單張推論，大幅提升 GPU 使用率。
        """
        detect_results = []
        total = len(image_paths)
        logger.info(f"即將進行 YOLO 批次分析，共 {total} 張圖片...")
        
        try:
            # 批次推論：一次將所有圖片路徑送入模型，避免逐張傳輸 CPU-GPU 瓶頸
            results = self.predict(image_paths, verbose=False)
            
            for idx, result in enumerate(results):
                try:
                    name = image_paths[idx] if idx < len(image_paths) else f"image_{idx}"
                    box_count = len(result.boxes)
                    conf = result.boxes.conf.max().item() if box_count > 0 else 0.0
                    detect_results.append((name, conf, box_count))
                except Exception as e:
                    logger.warning(f"YOLO 結果解析失敗 (index {idx}): {e}")
                    detect_results.append((image_paths[idx] if idx < len(image_paths) else f"image_{idx}", 0.0, 0))
                    
                if progress_callback:
                    progress_callback(idx + 1, total)
                    
        except Exception as e:
            logger.error(f"YOLO 批次推論失敗: {e}，降級為逐張處理")
            # 降級處理：若批次失敗，逐張處理
            from tqdm import tqdm
            for idx, name in enumerate(tqdm(image_paths, desc="YOLO 分析進度")):
                try:
                    results_iter = self.predict(name, verbose=False)
                    for r in results_iter:
                        box_count = len(r.boxes)
                        conf = r.boxes.conf.max().item() if box_count > 0 else 0.0
                        detect_results.append((name, conf, box_count))
                except Exception as e:
                    logger.warning(f"YOLO 推論失敗 {name}: {e}")
                    
                if progress_callback:
                    progress_callback(idx + 1, total)
                
        try:
            strategy = SampleStrategy(sample_way.lower())
        except ValueError:
            logger.warning(f"未知的 sample_way: {sample_way}，預設使用 negative")
            strategy = SampleStrategy.NEGATIVE
            
        if strategy == SampleStrategy.POSITIVE:
            detect_results.sort(key=lambda x: x[1])
        elif strategy == SampleStrategy.NEGATIVE:
            detect_results.sort(key=lambda x: x[1], reverse=True)
            
        return detect_results

    def get_confidence(self, image_path: str) -> float:
        try:
            results = self.predict(image_path, verbose=False)
            for r in results:
                if len(r.boxes) > 0:
                    return r.boxes.conf.max().item()
        except Exception as e:
            logger.warning(f"YOLO 推論失敗 {image_path}: {e}")
        return 0.0
        
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(model_path='{self._model_path}')>"

def path_check(path: str) -> bool:
    """檢查路徑是否為支援的圖片格式"""
    return is_supported_image_file(path)
