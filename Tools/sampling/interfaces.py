from abc import ABC, abstractmethod
from typing import List, Any, Optional
import numpy as np

class BaseDataset(ABC):
    """資料集抽象基底類別"""
    @abstractmethod
    def get_sources(self) -> List[str]:
        """取得資料來源列表"""
        pass

class FeatureExtractorInterface(ABC):
    """特徵提取器抽象介面"""
    @property
    @abstractmethod
    def device(self) -> Any:
        """取得使用的設備 (cpu/cuda)"""
        pass

    @abstractmethod
    def extract_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """提取單張影像的特徵向量"""
        pass

    @abstractmethod
    def extract_features_from_paths(self, image_paths: List[str]) -> np.ndarray:
        """批次提取影像特徵向量"""
        pass

class ObjectDetectorInterface(ABC):
    """物件偵測器抽象介面"""
    @abstractmethod
    def predict(self, source: Any, **kwargs) -> Any:
        """執行模型預測"""
        pass

    @abstractmethod
    def analyze(self, image_paths: List[str], sample_way: str) -> List[tuple]:
        """分析影像列表並根據策略排序"""
        pass
