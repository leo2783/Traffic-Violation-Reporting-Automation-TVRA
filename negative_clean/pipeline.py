import os
import cv2
import time
import numpy as np
import shutil
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
from scipy.stats import zscore
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from ultralytics import YOLO
import yt_dlp

from auto_label_classifier import AutoLabelClassifier

# ==========================================
# 1. 資料獲取模組 (Dataset Classes)
# ==========================================
class BaseDataset:
    def __init__(self):
        pass
    
    def get_sources(self):
        """回傳影片路徑或網址的列表"""
        raise NotImplementedError

class YoutubeDataset(BaseDataset):
    def __init__(self, target_count=200):
        super().__init__()
        self.target_count = target_count
        self.url = 'https://www.youtube.com/@WoWtchout/videos'
        
    def get_sources(self):
        options = webdriver.ChromeOptions()
        options.add_experimental_option("detach", False)
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        # 可開啟 headless 以在背景執行
        # options.add_argument("--headless")
        driver = webdriver.Chrome(options=options, service=Service(ChromeDriverManager().install()))
        driver.get(self.url)
        driver.maximize_window()
        
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, 'content'))
        )
        print("Start fetching YouTube links...")
        video_links = set()
        
        while len(video_links) < self.target_count:
            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(2)
            links = driver.execute_script(
                'return Array.from(document.querySelectorAll("a[href*=\\"/watch?v=\\"]")).map(a => a.href);'
            )
            for link in links:
                clean_link = link.split('&')[0]
                video_links.add(clean_link)
                if len(video_links) >= self.target_count:
                    break
            print(f"Currently fetched: {len(video_links)} videos")
            
        driver.quit()
        return list(video_links)[:self.target_count]

class LocalDataset(BaseDataset):
    def __init__(self, directory):
        super().__init__()
        self.directory = directory
        
    def get_sources(self):
        supported_formats = ('.mp4', '.avi', '.mov', '.ts', '.jpg', '.jpeg', '.png')
        sources = []
        if os.path.exists(self.directory):
            for f in os.listdir(self.directory):
                if f.lower().endswith(supported_formats):
                    sources.append(os.path.join(self.directory, f))
        return sources

# ==========================================
# 2. 特徵轉換模組 (Feature Extractor)
# ==========================================
class FeatureExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"FeatureExtractor 使用設備: {self.device}")
        # 使用 mobilenet_v3_small
        self.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.model.classifier = torch.nn.Identity() # 移除最後的分類層以取得特徵
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, image_paths):
        """將圖片路徑列表轉換為高維度特徵向量，分批讀取避免記憶體耗盡"""
        features = []
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i+batch_size]
                batch_tensors = []
                for p in batch_paths:
                    img = cv2.imread(p)
                    if img is None:
                        continue
                    # BGR 轉 RGB
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    tensor = self.transform(img_rgb)
                    batch_tensors.append(tensor)
                
                if not batch_tensors:
                    continue
                    
                batch_tensor = torch.stack(batch_tensors).to(self.device)
                out = self.model(batch_tensor)
                features.append(out.cpu().numpy())
                
        if features:
            return np.vstack(features)
        return np.array([])

# ==========================================
# 3. 核心自動化流水線 (Pipeline)
# ==========================================
class SamplingPipeline:
    def __init__(self, model_path, output_dir):
        # 因為腳本在 negative_clean 下運行，output_dir 直接用當前目錄 (.)
        self.model_path = model_path
        self.output_dir = output_dir
        self.auto_labeled_dir = os.path.join(self.output_dir, "auto_labeled")
        self.auto_images_dir = os.path.join(self.auto_labeled_dir, "images")
        self.auto_labels_dir = os.path.join(self.auto_labeled_dir, "labels")
        self.negative_sample_dir = os.path.join(self.output_dir, "negative_sample")
        
        self.temp_negative_dir = os.path.join(self.output_dir, "temp_negative_candidates")
        self.temp_auto_dir = os.path.join(self.output_dir, "temp_auto_candidates")
        
        self.setup_directories()
        
        print(f"Loading YOLO model from {self.model_path}...")
        self.yolo_model = YOLO(self.model_path)
        self.feature_extractor = FeatureExtractor()

    def setup_directories(self):
        """確保輸出目錄結構存在"""
        os.makedirs(self.auto_images_dir, exist_ok=True)
        os.makedirs(self.auto_labels_dir, exist_ok=True)
        os.makedirs(self.negative_sample_dir, exist_ok=True)
        os.makedirs(self.temp_negative_dir, exist_ok=True)
        os.makedirs(self.temp_auto_dir, exist_ok=True)

    def process_frame_results(self, results, frame, base_filename, negative_candidates, auto_candidates, frame_count, last_negative_frame, last_auto_frame):
        saved_count = 0
        for r in results:
            if len(r.boxes) == 0:
                continue
                
            confs = r.boxes.conf.cpu().numpy()
            if len(confs) == 0:
                continue
                
            max_conf = float(max(confs))
            
            # --- 信心度分類邏輯 ---
            if max_conf >= 0.8:
                # 1. > 0.8 儲存為 auto_labeled 候選，並維持 3 秒間隔防呆
                if frame_count - last_auto_frame >= 15:
                    img_path = os.path.join(self.temp_auto_dir, f"{base_filename}.jpg")
                    cv2.imwrite(img_path, frame)
                    
                    txt_path = os.path.join(self.temp_auto_dir, f"{base_filename}.txt")
                    boxes_to_save = []
                    xywhn = r.boxes.xywhn.cpu().numpy()
                    cls_indices = r.boxes.cls.cpu().numpy()
                    
                    for i in range(len(confs)):
                        if confs[i] >= 0.8: # 過濾單一框的信心度
                            c = int(cls_indices[i])
                            x, y, w, h = xywhn[i]
                            boxes_to_save.append(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                            
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        if boxes_to_save:
                            f.write('\n'.join(boxes_to_save))
                    
                    auto_candidates.append({
                        'path': img_path,
                        'txt_path': txt_path,
                        'max_conf': max_conf,
                        'name': base_filename
                    })
                    last_auto_frame = frame_count
                    saved_count += 1
                
            elif 0.2 < max_conf < 0.65:
                # 2. 0.2 ~ 0.65 作為 negative_sample 候選
                # 加入時間間隔限制 (相隔至少 15 個處理幀，約 3 秒)，確保樣本多樣性
                if frame_count - last_negative_frame >= 15:
                    cand_path = os.path.join(self.temp_negative_dir, f"{base_filename}.jpg")
                    cv2.imwrite(cand_path, frame)
                    negative_candidates.append({
                        'path': cand_path,
                        'max_conf': max_conf,
                        'name': base_filename
                    })
                    last_negative_frame = frame_count
        return saved_count, last_negative_frame, last_auto_frame

    def process_dataset(self, dataset: BaseDataset):
        """推論與擷取環節"""
        sources = dataset.get_sources()
        print(f"Found {len(sources)} sources to process.")
        
        negative_candidates = [] 
        auto_candidates = []
        
        for source_idx, source in enumerate(sources):
            print(f"\nProcessing source {source_idx + 1}/{len(sources)}: {source}")
            
            is_youtube = type(source) is str and ("youtu" in source or "http" in source)
            is_image = type(source) is str and source.lower().endswith(('.jpg', '.jpeg', '.png'))
            
            if is_image:
                print("Processing single image...")
                frame = cv2.imread(source)
                if frame is None:
                    continue
                results = self.yolo_model.predict(source=frame, conf=0.2, verbose=False)
                file_basename = os.path.splitext(os.path.basename(source))[0]
                base_filename = f"{file_basename}_img"
                saved, _, _ = self.process_frame_results(results, frame, base_filename, negative_candidates, auto_candidates, 15, 0, 0)
                print(f"Image completed. Auto-labeled {saved} frames.")
                continue

            # 若是影片，利用 OpenCV 抽幀
            video_path = source
            if is_youtube:
                print("Extracting direct stream URL using yt-dlp...")
                try:
                    # 使用 yt-dlp 解析串流位置，避免 YOLO 內建的 pafy 出現 "Waiting for stream 0"
                    ydl_opts = {
                        'format': 'best', 
                        'quiet': True, 
                        'no_warnings': True, 
                        'noplaylist': True
                    }
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(source, download=False)
                        video_path = info.get('url', source)
                except Exception as e:
                    print(f"yt-dlp extraction failed: {e}")
                    continue
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Failed to open video source: {source}")
                continue
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or np.isnan(fps):
                fps = 30.0 # 預設 30 fps
                
            # 每秒抽 5 幀 (5 fps) 的間隔
            frame_stride = max(1, int(round(fps / 5.0)))
            print(f"Video FPS: {fps:.2f}, extracting 1 frame every {frame_stride} frames (approx 5 fps)")
            
            frame_count = 0
            processed_count = 0
            saved_count = 0
            last_negative_frame = -999
            last_auto_frame = -999
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % frame_stride == 0:
                    results = self.yolo_model.predict(source=frame, conf=0.2, verbose=False)
                    
                    if is_youtube:
                        base_filename = f"youtube_{source_idx}_frame_{frame_count}"
                    else:
                        file_basename = os.path.splitext(os.path.basename(source))[0]
                        base_filename = f"{file_basename}_frame_{frame_count}"
                        
                    saved, last_negative_frame, last_auto_frame = self.process_frame_results(
                        results, frame, base_filename, negative_candidates, auto_candidates, processed_count, last_negative_frame, last_auto_frame
                    )
                    saved_count += saved
                    processed_count += 1
                    
                frame_count += 1
            
            cap.release()
            print(f"Source completed. Collected {saved_count} auto-labeled candidates.")
            
        print(f"\nTotal negative candidates collected: {len(negative_candidates)}")
        print(f"Total auto_labeled candidates collected: {len(auto_candidates)}")
        
        # 處理 negative_sample 分群與抽樣
        if len(negative_candidates) > 0:
            self.cluster_and_sample_negative(negative_candidates)
        else:
            print("No negative candidates found.")
            
        # 處理 auto_labeled 分群與前 10 名選取
        if len(auto_candidates) > 0:
            auto_classifier = AutoLabelClassifier(self.feature_extractor, self.auto_images_dir, self.auto_labels_dir)
            auto_classifier.cluster_and_select(auto_candidates, top_k=10)
        else:
            print("No auto_labeled candidates found.")
            
        # 清除暫存區
        if os.path.exists(self.temp_negative_dir):
            shutil.rmtree(self.temp_negative_dir)
        if os.path.exists(self.temp_auto_dir):
            shutil.rmtree(self.temp_auto_dir)
            
        print("Pipeline execution completed! 成果已保存至目前目錄的 auto_labeled 與 negative_sample 中。")


    def cluster_and_sample_negative(self, candidates):
        """K-Means 分群、定義難度指標與 Z-score 機率抽樣 (Negative Sample)"""
        print("\n--- Processing Negative Samples ---")
        print("Extracting features for negative candidates...")
        image_paths = [c['path'] for c in candidates]
        features = self.feature_extractor.extract_features(image_paths)
        
        print("Performing K-Means clustering...")
        # 決定群組數量，最多分10群
        n_clusters = min(10, len(candidates)) 
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features)
        
        # 依據群組整理候選資料的信心度
        cluster_confs = {i: [] for i in range(n_clusters)}
        for idx, cluster_id in enumerate(clusters):
            cluster_confs[cluster_id].append(candidates[idx]['max_conf'])
            
        # 定義難度指標：群集內的平均信心度
        cluster_metrics = {}
        for i in range(n_clusters):
            if cluster_confs[i]:
                cluster_metrics[i] = np.mean(cluster_confs[i])
            else:
                cluster_metrics[i] = 0.0
                
        metrics_list = [cluster_metrics[i] for i in range(n_clusters)]
        print(f"Cluster metrics (Difficulty indicator by mean conf): {metrics_list}")
        
        # 計算 Z-score 並轉換為機率權重 w
        if n_clusters > 1 and np.std(metrics_list) > 0:
            z_scores = zscore(metrics_list)
            # 轉換為大於 0 的權重 (類似 softmax)
            exp_z = np.exp(z_scores)
            weights = exp_z / np.sum(exp_z)
        else:
            weights = np.ones(n_clusters) / n_clusters
            
        print(f"Sampling weights w by cluster: {weights}")
        
        # 決定抽取數量 (此處設定為候選數量的 30%，並加上總數上限以確保多樣性不被單一影片淹沒)
        n_samples = min(1000, max(1, int(len(candidates) * 0.3)))
        
        # 分配機率給每一張照片
        cand_probs = []
        for idx, cluster_id in enumerate(clusters):
            cluster_size = len(cluster_confs[cluster_id])
            prob = weights[cluster_id] / cluster_size
            cand_probs.append(prob)
            
        cand_probs = np.array(cand_probs)
        cand_probs /= cand_probs.sum() # 正規化確保總和為 1
        
        # 根據權重 w 隨機抽取 negative sample
        sampled_indices = np.random.choice(
            len(candidates), 
            size=n_samples, 
            replace=False, 
            p=cand_probs
        )
        
        print(f"Saving {n_samples} negative samples...")
        for idx in sampled_indices:
            cand = candidates[idx]
            dest_path = os.path.join(self.negative_sample_dir, f"{cand['name']}.jpg")
            if os.path.exists(cand['path']):
                shutil.copy(cand['path'], dest_path)


# ==========================================
# 4. 程式進入點
# ==========================================
def main():
    print("=== Traffic Violation Reporting Automation (TVRA) ===")
    print("請選擇資料來源模式:")
    print("1. YouTube 影片串流")
    print("2. 本地影片或照片")
    choice = input("輸入選項 (1 或 2): ").strip()
    
    if choice == '1':
        num = input("請輸入欲處理的 YouTube 影片數量 (預設 200): ").strip()
        num = int(num) if num.isdigit() else 200
        dataset = YoutubeDataset(target_count=num)
    elif choice == '2':
        # 因為此腳本預期在 negative_clean 內執行，預設上一層的 test_video
        default_dir = os.path.join("..", "test_video")
        local_dir = input(f"請輸入本地資料夾路徑 (預設: {default_dir}): ").strip()
        if not local_dir:
            local_dir = default_dir
            if not os.path.exists(local_dir):
                os.makedirs(local_dir, exist_ok=True)
                print(f"已自動創建 {local_dir}，請將影片放入該資料夾後重新執行。")
                return
        dataset = LocalDataset(local_dir)
    else:
        print("無效選擇，結束程式。")
        return

    # 指定推論模型路徑 (相對專案根目錄)
    engine_path = os.path.join("YOLO_V4_Result", "train", "weights", "best.engine")
    pt_path = os.path.join("YOLO_V4_Result", "train", "weights", "best.pt")
    
    if os.path.exists(engine_path):
        model_path = engine_path
    elif os.path.exists(pt_path):
        print(f"找不到 {engine_path}，使用 {pt_path} 作為替代方案。")
        model_path = pt_path
    else:
        print(f"找不到 {engine_path} 或 {pt_path}。")
        model_path = input("請輸入模型路徑 (例如: ../yolo26n.pt): ").strip()
        if not model_path:
            return

    # 建立 Pipeline 並執行
    # 因腳本位於 negative_clean，我們把 output_dir 設為當前目錄 (.)
    pipeline = SamplingPipeline(model_path=model_path, output_dir=".")
    pipeline.process_dataset(dataset)

if __name__ == "__main__":
    main()
