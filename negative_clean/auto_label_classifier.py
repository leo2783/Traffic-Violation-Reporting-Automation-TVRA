import os
import shutil
from sklearn.cluster import KMeans

class AutoLabelClassifier:
    """
    獨立的 Auto Labeled 分類器，用於將高信心度的候選照片進行 K-Means 分群，
    並從每個群集中挑選出信心度最高的前 N 名照片，以確保樣本的場景多樣性。
    """
    def __init__(self, feature_extractor, auto_images_dir, auto_labels_dir):
        """
        初始化分類器
        :param feature_extractor: 用來抽取高維度特徵的物件 (需實作 extract_features 方法)
        :param auto_images_dir: 最終篩選出之照片的儲存目錄
        :param auto_labels_dir: 最終篩選出之標註檔的儲存目錄
        """
        self.feature_extractor = feature_extractor
        self.auto_images_dir = auto_images_dir
        self.auto_labels_dir = auto_labels_dir

    def cluster_and_select(self, candidates, top_k=5):
        """
        K-Means 分群並選取每群信心度前 N 名 (Auto Labeled)
        :param candidates: 包含候選資料資訊的字典列表，格式需包含:
                           - 'path': 影像暫存路徑
                           - 'txt_path': 標註檔暫存路徑
                           - 'max_conf': 該影像的最高信心度
                           - 'name': 檔案基準名稱
        :param top_k: 每個群集取前幾名 (預設 10)
        """
        print("\n--- Processing Auto Labeled Samples ---")
        if not candidates:
            print("No auto_labeled candidates found.")
            return

        print("Extracting features for auto_labeled candidates...")
        image_paths = [c['path'] for c in candidates]
        features = self.feature_extractor.extract_features(image_paths)
        
        print("Performing K-Means clustering for auto_labeled...")
        # 決定群組數量，最多分 10 群
        n_clusters = min(30, len(candidates)) 
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
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
                dest_img = os.path.join(self.auto_images_dir, f"{cand['name']}.jpg")
                shutil.copy(cand['path'], dest_img)
            # 複製標註檔
            if 'txt_path' in cand and os.path.exists(cand['txt_path']):
                dest_txt = os.path.join(self.auto_labels_dir, f"{cand['name']}.txt")
                shutil.copy(cand['txt_path'], dest_txt)
