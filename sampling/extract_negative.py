import os
import shutil
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image, ImageOps
from ultralytics import YOLO
import umap
import hdbscan

class NegativeSampler:
    """
    負樣本抽樣工具
    將去重後的圖片透過特徵擷取、降維分群，再依據群體平均 YOLO 信心度轉換為抽取機率，
    以機率決定最終抽取的樣本，確保信心度越高越容易被抽取。
    """
    def __init__(self, yolo_weights=r"C:/Users/qet63/Documents/Traffic-Violation-Reporting-Automation-TVRA-/YOLO_V4_Result/train/weights/best.pt"):
        # 保護內部屬性，避免被惡意修改
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._yolo_weights = yolo_weights
        self._init_models()

    def _init_models(self):
        """初始化特徵擷取模型與信心度預測模型"""
        # 1. 載入 MobileNetV3 (如同 embedding.py)
        self._weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        self._embed_model = models.mobilenet_v3_small(weights=self._weights)
        self._embed_model.classifier = torch.nn.Identity()
        self._embed_model.to(self._device)
        self._embed_model.eval()
        
        # 設定影像前處理
        self._preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 2. 載入 YOLO 模型
        self._yolo_model = YOLO(self._yolo_weights, task="detect")

    def _extract_embedding(self, image_path):
        """讀取單張圖片，產出 1D 特徵向量"""
        img = Image.open(image_path).convert("RGB")
        img = ImageOps.pad(img, (224, 224), color=(0, 0, 0), centering=(0.5, 0.5))
        input_tensor = self._preprocess(img).unsqueeze(0).to(self._device)
        
        with torch.no_grad():
            embedding = self._embed_model(input_tensor)
            
        return embedding.squeeze().cpu().numpy()

    def _get_confidence(self, image_path):
        """獲取圖片中物件的最大信心度"""
        results = self._yolo_model.predict(image_path, verbose=False)
        if len(results[0].boxes) > 0:
            return results[0].boxes.conf.max().item()
        return 0.0

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
        # 加入溫度參數以放大信心度差異
        temperature = 5.0
        exp_scores = np.exp(scores * temperature)
        probabilities = exp_scores / np.sum(exp_scores)
        
        return probabilities

    def sample(self, image_paths: list, num_samples: int, output_dir: str = "negative_result"):
        """
        執行抽樣流程並將結果儲存。
        
        Args:
            image_paths: 強制輸入，要進行抽樣的圖片路徑清單。
            num_samples: 強制輸入，要抽取的樣本數。
            output_dir: 選擇性輸入，輸出的資料夾路徑，預設為 "negative_result"。
            
        Returns:
            list: 被抽取的檔案路徑清單
        """
        if not image_paths:
            raise ValueError("輸入的圖片路徑清單 (image_paths) 不可為空")
        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError("抽樣數量 (num_samples) 必須為大於 0 的整數")
            
        if num_samples > len(image_paths):
            print(f"[警告] 要求抽樣數 {num_samples} 大於總圖片數 {len(image_paths)}，將抽取所有有效圖片。")
            num_samples = len(image_paths)

        print(f"開始處理 {len(image_paths)} 張圖片...")
        
        embeddings = []
        confidences = []
        valid_paths = []
        
        for path in image_paths:
            if os.path.exists(path):
                emb = self._extract_embedding(path)
                conf = self._get_confidence(path)
                embeddings.append(emb)
                confidences.append(conf)
                valid_paths.append(path)
            else:
                print(f"[警告] 找不到檔案: {path}")
                
        if not valid_paths:
            raise ValueError("沒有找到任何有效的圖片檔案進行抽樣")
            
        # 更新實際可抽樣數量
        actual_num_samples = min(num_samples, len(valid_paths))
            
        embeddings_matrix = np.array(embeddings)
        
        print("進行 UMAP 降維與 HDBSCAN 分群...")
        # 防呆機制：樣本數過少時跳過複雜分群
        if len(valid_paths) < 5:
            print("[提示] 樣本數過少，跳過分群，直接計算個體信心度機率。")
            labels = np.zeros(len(valid_paths))
        else:
            labels = self._reduce_and_cluster(embeddings_matrix)
            
        print("計算抽樣機率...")
        probabilities = self._calculate_probabilities(labels, confidences)
        
        print(f"根據機率隨機抽取 {actual_num_samples} 張圖片...")
        # 根據機率隨機抽取 (不重複)
        sampled_indices = np.random.choice(
            len(valid_paths), 
            size=actual_num_samples, 
            replace=False, 
            p=probabilities
        )
        
        sampled_paths = [valid_paths[i] for i in sampled_indices]
        
        # 儲存結果
        print(f"儲存抽樣結果至 '{output_dir}' ...")
        os.makedirs(output_dir, exist_ok=True)
        
        final_dest_paths = []
        for path in sampled_paths:
            file_name = os.path.basename(path)
            dest_path = os.path.join(output_dir, file_name)
            shutil.copy(path, dest_path)
            final_dest_paths.append(dest_path)
            
        print(f"成功！已將 {len(final_dest_paths)} 張圖片儲存至 {output_dir}")
        return final_dest_paths
