import os
import logging
from enum import Enum
from typing import List, Optional, Any
import numpy as np

import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image, ImageOps, UnidentifiedImageError

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

    def extract_features_from_paths(self, image_paths: List[str]) -> np.ndarray:
        """
        Batch processing: extract feature vectors for multiple images, return numpy matrix.
        批次處理：對多張圖片提取特徵向量，回傳 numpy 矩陣。
        """
        features = []
        for path in image_paths:
            emb = self.extract_embedding(path)
            if emb is not None:
                features.append(emb)
        if features:
            return np.vstack(features)
        return np.array([])
        
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(device={self._device})>"

class SampleStrategy(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"

class YoloAnalyzer(ObjectDetectorInterface):
    """YOLO 模型推論封裝"""
    def __init__(self, model_path: str) -> None:
        if not isinstance(model_path, str):
            raise TypeError("model_path 必須是字串")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到 YOLO 模型權重檔案: {model_path}")
            
        from ultralytics import YOLO
        self._model = YOLO(model_path, task="detect")
        self._model_path = model_path
        
    def predict(self, source: Any, **kwargs) -> Any:
        return self._model.predict(source, **kwargs)

    def analyze(self, image_paths: List[str], sample_way: str) -> List[tuple]:
        from tqdm import tqdm
        detect_results = []
        logger.info(f"即將進行 YOLO 分析，共 {len(image_paths)} 張圖片...")
        for name in tqdm(image_paths, desc="YOLO 分析進度"):
            try:
                results = self.predict(name, verbose=False)
                box_count = len(results[0].boxes)
                conf = results[0].boxes.conf.max().item() if box_count > 0 else 0.0
                detect_results.append((name, conf, box_count))
            except Exception as e:
                logger.warning(f"YOLO 推論失敗 {name}: {e}")
                
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
            if len(results[0].boxes) > 0:
                return results[0].boxes.conf.max().item()
        except Exception as e:
            logger.warning(f"YOLO 推論失敗 {image_path}: {e}")
        return 0.0
        
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(model_path='{self._model_path}')>"

def path_check(path: str) -> bool:
    """檢查路徑是否為支援的圖片格式"""
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
    if os.path.exists(path):
        return path.lower().endswith(supported_formats)
    return False
