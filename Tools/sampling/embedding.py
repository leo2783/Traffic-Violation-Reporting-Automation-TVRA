import numpy as np
import torch
import os
import re
import shutil
import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Callable
from utils import FeatureExtractor, YoloAnalyzer, path_check, get_device

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

    def process_batch(self, image_paths: List[str], use_confidence: bool = False, sample_way: str = "positive",
                      write_mode: str = "per-folder", output_folder: str = "",
                      progress_callback: Optional[Callable[[int, int], None]] = None) -> List[str]:
        """接收一串圖片路徑，回傳不重複的路徑清單
        
        :param progress_callback: 進度回呼，格式 callback(current_step, total_steps)
        """
        if use_confidence and not self._yolo_analyzer:
            raise ValueError("必須提供 yolo_weights 才能使用 YOLO 信心度進行排序")

        embeddings = []
        box_counts = []
        total_steps = 100
        
        if use_confidence:
            scored_paths = self._yolo_analyzer.analyze(image_paths=image_paths, sample_way=sample_way,
                                                       progress_callback=progress_callback)
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
            if progress_callback:
                callback_step = (i + 1) + len(image_paths)
                progress_callback(callback_step, total_steps)
        
        if not embeddings:
            logger.warning("沒有有效的圖片特徵，結束處理。")
            return []

        # 轉換為 PyTorch Tensor 以進行高速矩陣運算
        embeddings_matrix = torch.tensor(np.array(embeddings), device=self._feature_extractor.device)
        
        logger.info("開始計算全局相似度矩陣...")
        # 一次性計算出所有要保留的 index，確保 PyTorch 批次運算效能最大化
        keep_indices = self._calculate_duplicates(embeddings_matrix, valid_box_counts)
        
        logger.info(f"相似度計算完成，開始執行 '{write_mode}' 寫入策略...")
        final_keep_paths = []
        
        if write_mode == "per-frame":
            # 逐幀模式：立刻按順序拷貝並輸出日誌
            for idx in keep_indices:
                path = valid_paths[idx]
                final_keep_paths.append(path)
                if output_folder:
                    file_name = Path(path).name
                    dest_path = Path(output_folder) / file_name
                    shutil.copy(path, dest_path)
                    logger.info(f"[Per-Frame] 已寫入不重複圖片: {file_name}")
                    
        elif write_mode == "per-video":
            # 逐影片模式：根據檔名前綴分組 (容錯解析)
            video_groups = defaultdict(list)
            for idx in keep_indices:
                path = valid_paths[idx]
                file_name = Path(path).name
                
                # 嘗試使用正則表達式或字串分割來安全地找出影片名
                # 假設檔名如: 2026_0430_010931_001A_frame_001.jpg
                match = re.match(r"(.+)_frame_\d+\..+", file_name, re.IGNORECASE)
                if match:
                    video_name = match.group(1)
                elif "_frame_" in file_name:
                    video_name = file_name.split("_frame_")[0]
                else:
                    video_name = "unknown_video"
                    
                video_groups[video_name].append(path)
                
            for video_name, paths in video_groups.items():
                logger.info(f"正在寫入影片群組: {video_name} (共 {len(paths)} 張唯一的圖片)")
                for path in paths:
                    final_keep_paths.append(path)
                    if output_folder:
                        file_name = Path(path).name
                        dest_path = Path(output_folder) / file_name
                        shutil.copy(path, dest_path)
                
                if output_folder:
                    logger.info(f"[Per-Video] 影片 {video_name} 寫入完畢。")
                    
        else: # per-folder
            # 全部處理完再一次性寫入
            for idx in keep_indices:
                path = valid_paths[idx]
                final_keep_paths.append(path)
                if output_folder:
                    file_name = Path(path).name
                    dest_path = Path(output_folder) / file_name
                    shutil.copy(path, dest_path)
            
            if output_folder:
                logger.info(f"[Per-Folder] 全部結果已一次性寫入 ({len(keep_indices)} 張)")
        
        if progress_callback:
            progress_callback(total_steps, total_steps)
        
        return final_keep_paths
        
    def _calculate_duplicates(self, embeddings_matrix: torch.Tensor, items: List[int]) -> np.ndarray:
        """核心矩陣運算 (使用 PyTorch 加速)"""
        norms = torch.norm(embeddings_matrix, dim=1, keepdim=True) + 1e-10
        normalized = embeddings_matrix / norms
        sim_matrix = torch.mm(normalized, normalized.t())
        
        if not items:
            logger.info("未啟用標註數量比對邏輯。")
        else:
            counts_arr = torch.tensor(items, device=self._feature_extractor.device)
            same_box_matrix = counts_arr.unsqueeze(1) == counts_arr.unsqueeze(0)
            sim_matrix = sim_matrix * same_box_matrix.float()
            
        upper_tri = torch.triu(sim_matrix, diagonal=1)
        duplicates = torch.any(upper_tri > self._threshold, dim=0)
        duplicates = duplicates.cpu().numpy()
        return np.where(~duplicates)[0]
