import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image, ImageOps, UnidentifiedImageError
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def safe_image_open(image_path):
    """安全地讀取圖片，若發生損毀則回傳 None"""
    try:
        img = Image.open(image_path).convert("RGB")
        return img
    except (UnidentifiedImageError, OSError) as e:
        logging.warning(f"無法讀取圖片 {image_path}: {e}")
        return None

class FeatureExtractor:
    """MobileNetV3 影像特徵擷取器"""
    def __init__(self):
        self.device = get_device()
        self._weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        self.model = models.mobilenet_v3_small(weights=self._weights)
        self.model.classifier = torch.nn.Identity()
        self.model.to(self.device)
        self.model.eval()
        
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_embedding(self, image_path):
        """讀取單張圖片，產出 1D 特徵向量 (回傳 numpy array)"""
        img = safe_image_open(image_path)
        if img is None:
            return None
            
        img = ImageOps.pad(img, (224, 224), color=(0, 0, 0), centering=(0.5, 0.5))
        input_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model(input_tensor)
            
        return embedding.squeeze().cpu().numpy()

class YoloAnalyzer:
    """YOLO 模型推論封裝"""
    def __init__(self, model_path):
        from ultralytics import YOLO
        self.model = YOLO(model_path, task="detect")
        
    def analyze(self, image_paths: list, sample_way: str):
        from tqdm import tqdm
        detect_results = []
        logging.info(f"即將進行 YOLO 分析，共 {len(image_paths)} 張圖片...")
        for name in tqdm(image_paths, desc="YOLO 分析進度"):
            try:
                results = self.model.predict(name, verbose=False)
                box_count = len(results[0].boxes)
                conf = results[0].boxes.conf.max().item() if box_count > 0 else 0.0
                detect_results.append((name, conf, box_count))
            except Exception as e:
                logging.warning(f"YOLO 推論失敗 {name}: {e}")
                
        if sample_way == "positive":
            detect_results.sort(key=lambda x: x[1])
        elif sample_way == "negative":
            detect_results.sort(key=lambda x: x[1], reverse=True)
        return detect_results

    def get_confidence(self, image_path):
        try:
            results = self.model.predict(image_path, verbose=False)
            if len(results[0].boxes) > 0:
                return results[0].boxes.conf.max().item()
        except Exception as e:
            logging.warning(f"YOLO 推論失敗 {image_path}: {e}")
        return 0.0

def path_check(path):
    """檢查路徑是否為支援的圖片格式"""
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
    if os.path.exists(path):
        return path.lower().endswith(supported_formats)
    return False
