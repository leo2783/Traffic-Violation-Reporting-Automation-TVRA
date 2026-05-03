import torch
import numpy as np
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image,ImageOps
import os
from ultralytics import YOLO
from tqdm import tqdm

class YoloAnalyzer:
    def __init__(self, model_path="C:/Users/qet63/Documents/Traffic-Violation-Reporting-Automation-TVRA-/YOLO_V4_Result/train/weights/best.pt"):
        self._model = YOLO(model_path, task="detect")
        
    def analyze(self, image_paths: list, sample_way: str):
        detect_results = []
        print(f"即將進行 YOLO 分析，共 {len(image_paths)} 張圖片...")
        for name in tqdm(image_paths, desc="YOLO 分析進度"):
            results = self._model.predict(name, verbose=False)
            box_count = len(results[0].boxes)
            if box_count > 0:
                conf = results[0].boxes.conf.max().item()
            else:
                conf = 0.0
            detect_results.append((name, conf, box_count))
            
        if sample_way == "positive":
            detect_results.sort(key=lambda x: x[1])
        elif sample_way == "negative":
            detect_results.sort(key=lambda x: x[1], reverse=True)
        return detect_results

class ImageDeduplicator:
    def __init__(self, threshold=0.95):
        self._threshold = threshold
        # 1. 初始化並載入模型
        self._weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        self._yolo_analyzer = YoloAnalyzer(r"C:/Users/qet63/Documents/Traffic-Violation-Reporting-Automation-TVRA-/YOLO_V4_Result/train/weights/best.pt")
        self._model = models.mobilenet_v3_small(weights=self._weights)
        self._model.classifier = torch.nn.Identity()
        self._model.eval()
        self._preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _extract_embedding(self, image_path):
        """讀取單張圖片，產出 1D 特徵向量"""
        img = Image.open(image_path).convert("RGB")
        img = ImageOps.pad(img, (224, 224), color=(0, 0, 0), centering=(0.5, 0.5))
        input_tensor = self._preprocess(img).unsqueeze(0)
        
        with torch.no_grad():
            embedding = self._model(input_tensor)
            
        # 將 PyTorch Tensor 轉回 NumPy 陣列並攤平
        return embedding.squeeze().numpy()
    def _path_check(self, path):
        supported_formats = ('.jpg', '.jpeg', '.png')
        if os.path.exists(path):
            return path.lower().endswith(supported_formats)
        return False
    def process_batch(self, image_paths:list,confident=False,sample_way="positive"):
        """接收一串圖片路徑，回傳不重複的路徑清單"""
        # 1. 收集所有圖片的 Embedding
        embeddings = []
        box_counts = []
        if confident:
            scored_paths = self._yolo_analyzer.analyze(image_paths=image_paths, sample_way=sample_way)
            image_paths = [item[0] for item in scored_paths]
            box_counts = [item[2] for item in scored_paths]
        
        valid_paths = []
        valid_box_counts = []
        for i, path in enumerate(image_paths):
            if self._path_check(path):
                emb = self._extract_embedding(path)
                embeddings.append(emb)
                valid_paths.append(path)
                if box_counts: 
                    valid_box_counts.append(box_counts[i])
        
        # 將串列轉成形狀為 (N, D) 的 NumPy 矩陣
        embeddings_matrix = np.array(embeddings)
        keep_indices = self._calculate_duplicates(embeddings_matrix, valid_box_counts)
        final_keep_paths = [valid_paths[i] for i in keep_indices]
        
        return final_keep_paths
        
    def _calculate_duplicates(self, embeddings_matrix,items:list):
        """(私有方法) 核心矩陣運算"""
        norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True) + 1e-10
        normalized = embeddings_matrix / norms
        sim_matrix = np.dot(normalized, normalized.T)
        if not items:
            print("[警告] items 是空清單，略過標註數量比對邏輯。")
        else:
            counts_arr = np.array(items)
            same_box_matrix = counts_arr[:, None] == counts_arr[None, :] 
            # 數量不同則強制將相似度歸零
            sim_matrix = sim_matrix * same_box_matrix
        upper_tri = np.triu(sim_matrix, k=1)
        # 2. 只要同一直行 (axis=0) 中有任何一個相似度大於閾值，就標記為重複 (True)
        duplicates = np.any(upper_tri > self._threshold, axis=0)
        # 3. 回傳不重複 (False) 的索引陣列
        return np.where(~duplicates)[0]