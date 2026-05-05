"""
Traffic Violation Reporting Automation (TVRA) - Auto Label Classifier
Independent Auto Labeled classifier for clustering and selecting high-confidence samples.
獨立的 Auto Labeled 分類器，用於將高信心度的候選照片進行 K-Means 分群，
並從每個群集中挑選出信心度最高的前 N 名照片，以確保樣本的場景多樣性。
"""

import os
import shutil
from sklearn.cluster import KMeans
from utils import FeatureExtractor


from interfaces import FeatureExtractorInterface
from typing import List, Dict, Any

class AutoLabelClassifier:
    """
    獨立的 Auto Labeled 分類器，用於將高信心度的候選照片進行 K-Means 分群，
    並從每個群集中挑選出信心度最高的前 N 名照片，以確保樣本的場景多樣性。
    """
    def __init__(self, feature_extractor: FeatureExtractorInterface, auto_images_dir: str, auto_labels_dir: str, max_clusters: int = 30, random_state: int = 42):
        """
        初始化分類器
        :param feature_extractor: 用來抽取高維度特徵的物件 (需實作 FeatureExtractorInterface 介面)
        :param auto_images_dir: 最終篩選出之照片的儲存目錄
        :param auto_labels_dir: 最終篩選出之標註檔的儲存目錄
        :param max_clusters: K-Means 最大群組數量 (預設 30)
        :param random_state: K-Means 隨機數種子 (預設 42)
        """
        if not isinstance(feature_extractor, FeatureExtractorInterface):
            raise TypeError("feature_extractor 必須實作 FeatureExtractorInterface 介面")
            
        self._feature_extractor = feature_extractor
        self._auto_images_dir = auto_images_dir
        self._auto_labels_dir = auto_labels_dir
        self._max_clusters = max_clusters
        self._random_state = random_state

    def cluster_and_select(self, candidates: List[Dict[str, Any]], top_k: int = 5) -> None:
        """
        K-Means 分群並選取每群信心度前 N 名 (Auto Labeled)
        :param candidates: 包含候選資料資訊的字典列表
        :param top_k: 每個群集取前幾名 (預設 5)
        """
        print("\n--- Processing Auto Labeled Samples ---")
        if not candidates:
            print("No auto_labeled candidates found.")
            return

        print("Extracting features for auto_labeled candidates...")
        image_paths = [c['path'] for c in candidates]
        features = self._feature_extractor.extract_features_from_paths(image_paths)
        
        print("Performing K-Means clustering for auto_labeled...")
        # 決定群組數量，最大不超過 max_clusters
        n_clusters = min(self._max_clusters, len(candidates)) 
        kmeans = KMeans(n_clusters=n_clusters, random_state=self._random_state)
        clusters = kmeans.fit_predict(features)
        
        # 依據群組將候選資料分類
        cluster_groups = {i: [] for i in range(n_clusters)}
        for idx, cluster_id in enumerate(clusters):
            cluster_groups[cluster_id].append(candidates[idx])
            
        selected_candidates = []
        for i in range(n_clusters):
            group = cluster_groups[i]
            # 依照 max_conf 降冪排序，取前 top_k 名
            group.sort(key=lambda x: x['max_conf'], reverse=True)
            top_selection = group[:top_k]
            selected_candidates.extend(top_selection)
            
        print(f"Selected {len(selected_candidates)} top confidence frames across {n_clusters} clusters.")
        
        # 將被選中的資料搬移至最終儲存目錄
        for cand in selected_candidates:
            # 複製影像
            if os.path.exists(cand['path']):
                dest_img = os.path.join(self._auto_images_dir, f"{cand['name']}.jpg")
                shutil.copy(cand['path'], dest_img)
            # 複製標註檔
            if 'txt_path' in cand and os.path.exists(cand['txt_path']):
                dest_txt = os.path.join(self._auto_labels_dir, f"{cand['name']}.txt")
                shutil.copy(cand['txt_path'], dest_txt)
                
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(max_clusters={self._max_clusters}, random_state={self._random_state})>"
