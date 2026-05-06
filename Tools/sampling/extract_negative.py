import os
import shutil
import numpy as np
import logging
import umap
import hdbscan
try:
    from .utils import FeatureExtractor, YoloAnalyzer, path_check
except ImportError:
    from utils import FeatureExtractor, YoloAnalyzer, path_check

class NegativeSampler:
    """
    負樣本抽樣工具
    將去重後的圖片透過特徵擷取、降維分群，再依據群體平均 YOLO 信心度轉換為抽取機率，
    以機率決定最終抽取的樣本，確保信心度越高越容易被抽取。
    """
    def __init__(self, yolo_weights, temperature=5.0):
        self.temperature = temperature
        self.feature_extractor = FeatureExtractor()
        self.yolo_analyzer = YoloAnalyzer(yolo_weights)

    def _reduce_and_cluster(self, embeddings_matrix):
        """UMAP 降維並使用 HDBSCAN 進行分群"""
        n_samples = len(embeddings_matrix)
        # 動態調整 n_neighbors 以避免樣本數過少時報錯
        n_neighbors = min(15, n_samples - 1)
        if n_neighbors < 2:
            n_neighbors = 2
            
        # UMAP 降維
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings_matrix)
        
        # HDBSCAN 分群
        min_cluster_size = max(2, min(5, n_samples // 5))
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        labels = clusterer.fit_predict(reduced_embeddings)
        return labels

    def _calculate_probabilities(self, labels, confidences):
        """計算各群平均信心度，並轉換為抽取機率"""
        labels = np.array(labels)
        confidences = np.array(confidences)
        unique_labels = np.unique(labels)
        
        # 計算每群平均信心度 (包含 noise 分類 -1)
        group_avg_conf = {}
        for label in unique_labels:
            mask = (labels == label)
            group_avg_conf[label] = confidences[mask].mean()
            
        # 將每個樣本的基礎分數設為其所屬群集的平均信心度
        scores = np.zeros(len(confidences))
        for i, label in enumerate(labels):
            scores[i] = group_avg_conf[label]
            
        # 將信心度用數學方式轉換為機率 (使用 Softmax)
        exp_scores = np.exp(scores * self.temperature)
        probabilities = exp_scores / np.sum(exp_scores)
        
        return probabilities

    def sample(self, image_paths: list, num_samples: int, output_dir: str = "negative_result"):
        """執行抽樣流程並將結果儲存"""
        if not image_paths:
            raise ValueError("輸入的圖片路徑清單 (image_paths) 不可為空")
        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError("抽樣數量 (num_samples) 必須為大於 0 的整數")
            
        if num_samples > len(image_paths):
            logging.warning(f"要求抽樣數 {num_samples} 大於總圖片數 {len(image_paths)}，將抽取所有有效圖片。")
            num_samples = len(image_paths)

        logging.info(f"開始處理 {len(image_paths)} 張圖片...")
        
        embeddings = []
        confidences = []
        valid_paths = []
        
        for path in image_paths:
            if path_check(path):
                emb = self.feature_extractor.extract_embedding(path)
                if emb is not None:
                    conf = self.yolo_analyzer.get_confidence(path)
                    embeddings.append(emb)
                    confidences.append(conf)
                    valid_paths.append(path)
            else:
                logging.warning(f"不支援的檔案格式或檔案不存在: {path}")
                
        if not valid_paths:
            raise ValueError("沒有找到任何有效的圖片檔案進行抽樣")
            
        actual_num_samples = min(num_samples, len(valid_paths))
        embeddings_matrix = np.array(embeddings)
        
        logging.info("進行 UMAP 降維與 HDBSCAN 分群...")
        if len(valid_paths) < 5:
            logging.info("樣本數過少，跳過分群，直接計算個體信心度機率。")
            labels = np.zeros(len(valid_paths))
        else:
            labels = self._reduce_and_cluster(embeddings_matrix)
            
        logging.info("計算抽樣機率...")
        probabilities = self._calculate_probabilities(labels, confidences)
        
        logging.info(f"根據機率隨機抽取 {actual_num_samples} 張圖片...")
        sampled_indices = np.random.choice(
            len(valid_paths), 
            size=actual_num_samples, 
            replace=False, 
            p=probabilities
        )
        
        sampled_paths = [valid_paths[i] for i in sampled_indices]
        
        logging.info(f"儲存抽樣結果至 '{output_dir}' ...")
        os.makedirs(output_dir, exist_ok=True)
        
        final_dest_paths = []
        for path in sampled_paths:
            file_name = os.path.basename(path)
            dest_path = os.path.join(output_dir, file_name)
            shutil.copy(path, dest_path)
            final_dest_paths.append(dest_path)
            
        logging.info(f"成功！已將 {len(final_dest_paths)} 張圖片儲存至 {output_dir}")
        return final_dest_paths
