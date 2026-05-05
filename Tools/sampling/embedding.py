import numpy as np
import torch
import os
import logging
from utils import FeatureExtractor, YoloAnalyzer, path_check, get_device

from typing import List, Optional

logger = logging.getLogger(__name__)

class ImageDeduplicator:
    def __init__(self, threshold: float = 0.95, yolo_weights: Optional[str] = None):
        if not isinstance(threshold, float) or not (0.0 <= threshold <= 1.0):
            raise ValueError("threshold 必須是 0.0 到 1.0 之間的浮點數")
            
        self._threshold = threshold
        # 共用特徵擷取器
        self._feature_extractor = FeatureExtractor()
        # 初始化 YOLO (如果有指定)
        self._yolo_analyzer = YoloAnalyzer(yolo_weights) if yolo_weights else None

    def process_batch(self, image_paths: List[str], use_confidence: bool = False, sample_way: str = "positive") -> List[str]:
        """接收一串圖片路徑，回傳不重複的路徑清單"""
        if use_confidence and not self._yolo_analyzer:
            raise ValueError("必須提供 yolo_weights 才能使用 YOLO 信心度進行排序")

        embeddings = []
        box_counts = []
        
        if use_confidence:
            scored_paths = self._yolo_analyzer.analyze(image_paths=image_paths, sample_way=sample_way)
            image_paths = [item[0] for item in scored_paths]
            box_counts = [item[2] for item in scored_paths]
        
        valid_paths = []
        valid_box_counts = []
        
        logger.info("開始擷取特徵向量...")
        for i, path in enumerate(image_paths):
            if path_check(path):
                emb = self._feature_extractor.extract_embedding(path)
                if emb is not None:
                    embeddings.append(emb)
                    valid_paths.append(path)
                    if box_counts: 
                        valid_box_counts.append(box_counts[i])
        
        if not embeddings:
            logger.warning("沒有有效的圖片特徵，結束處理。")
            return []

        # 轉換為 PyTorch Tensor 以進行高速矩陣運算
        embeddings_matrix = torch.tensor(np.array(embeddings), device=self._feature_extractor.device)
        
        logger.info("開始計算相似度...")
        keep_indices = self._calculate_duplicates(embeddings_matrix, valid_box_counts)
        final_keep_paths = [valid_paths[i] for i in keep_indices]
        
        return final_keep_paths
        
    def _calculate_duplicates(self, embeddings_matrix: torch.Tensor, items: List[int]) -> np.ndarray:
        """核心矩陣運算 (使用 PyTorch 加速)"""
        # 進行 L2 正規化
        norms = torch.norm(embeddings_matrix, dim=1, keepdim=True) + 1e-10
        normalized = embeddings_matrix / norms
        
        # 計算餘弦相似度矩陣 (N x N)
        # 對於非常大的 N，這裡可能會 OOM，可以改用 batch 計算，但這裡先用 PyTorch 原生運算加速
        sim_matrix = torch.mm(normalized, normalized.t())
        
        if not items:
            logger.info("未啟用標註數量比對邏輯。")
        else:
            counts_arr = torch.tensor(items, device=self._feature_extractor.device)
            # 數量不同則將相似度歸零
            same_box_matrix = counts_arr.unsqueeze(1) == counts_arr.unsqueeze(0)
            sim_matrix = sim_matrix * same_box_matrix.float()
            
        # 取得上三角矩陣 (不包含對角線)
        upper_tri = torch.triu(sim_matrix, diagonal=1)
        
        # 只要同一直行中有任何一個相似度大於閾值，就標記為重複 (True)
        duplicates = torch.any(upper_tri > self._threshold, dim=0)
        
        # 轉回 numpy array
        duplicates = duplicates.cpu().numpy()
        
        # 回傳不重複 (False) 的索引陣列
        return np.where(~duplicates)[0]
