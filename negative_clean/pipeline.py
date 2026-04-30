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

# Import the independently implemented Auto Labeled Classifier
# 匯入獨立實作的 Auto Labeled 分類器
from auto_label_classifier import AutoLabelClassifier

"""
Traffic Violation Reporting Automation (TVRA) - Automated Sampling Pipeline / 自動化抽樣流水線
This script automates the extraction of image samples from YouTube streams or local videos,
performs YOLO inference, filters auto_labeled and negative_sample candidates by confidence,
and selects diverse, representative training samples through feature clustering and statistical sampling.
本腳本負責實現自動化從 YouTube 串流或本地影片中擷取影像樣本、進行 YOLO 推論、
根據信心度過濾 auto_labeled 與 negative_sample 候選、並透過特徵分群與統計抽樣
選出最具多樣性與代表性的訓練樣本。
"""

# ==========================================
# 1. Data Acquisition Module / 資料獲取模組 (Dataset Classes)
# ==========================================
class BaseDataset:
    """Base class for datasets, defining the interface for obtaining sources. / 資料集基礎類別，定義獲取來源的介面"""
    def __init__(self):
        pass
    
    def get_sources(self):
        """Should return a list of video paths or YouTube URLs. / 應回傳影片路徑或 YouTube 網址的列表"""
        raise NotImplementedError

class YoutubeDataset(BaseDataset):
    """
    YouTube Video Dataset Class
    Responsible for scraping video URLs from a specified channel using Selenium.
    YouTube 影片資料集類別
    負責使用 Selenium 爬取指定頻道的影片網址
    """
    def __init__(self, target_count=200):
        super().__init__()
        self.target_count = target_count
        self.url = 'https://www.youtube.com/@WoWtchout/videos'
        
    def get_sources(self):
        """Execute the scraper to obtain video links. / 執行爬蟲獲取影片連結"""
        options = webdriver.ChromeOptions()
        options.add_experimental_option("detach", False)
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        # Uncomment below to enable headless mode if you don't want to see the browser.
        # 若不想看到瀏覽器開啟，可取消下面註解開啟 headless 模式
        # options.add_argument("--headless")
        driver = webdriver.Chrome(options=options, service=Service(ChromeDriverManager().install()))
        driver.get(self.url)
        driver.maximize_window()
        
        # Wait for key elements to load
        # 等待頁面載入關鍵元素
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, 'content'))
        )
        print("Starting to fetch YouTube video links... / 開始擷取 YouTube 影片連結...")
        video_links = set()
        
        # Scroll page until enough links are collected
        # 循環捲動頁面直到收集足夠數量的連結
        while len(video_links) < self.target_count:
            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(2) # Wait for new content to load
            links = driver.execute_script(
                'return Array.from(document.querySelectorAll("a[href*=\\"/watch?v=\\"]")).map(a => a.href);'
            )
            for link in links:
                clean_link = link.split('&')[0] # Remove extra parameters
                video_links.add(clean_link)
                if len(video_links) >= self.target_count:
                    break
            print(f"Currently fetched: {len(video_links)} videos / 目前已收集: {len(video_links)} 部影片")
            
        driver.quit()
        return list(video_links)[:self.target_count]

class LocalDataset(BaseDataset):
    """
    Local Folder Dataset Class
    Responsible for scanning video or image files in a specified path.
    本地資料夾資料集類別
    負責掃描指定路徑下的影片或圖片檔案
    """
    def __init__(self, directory):
        super().__init__()
        self.directory = directory
        
    def get_sources(self):
        """Scan for supported formats in the folder. / 掃描資料夾內支援的格式"""
        supported_formats = ('.mp4', '.avi', '.mov', '.ts', '.jpg', '.jpeg', '.png')
        sources = []
        if os.path.exists(self.directory):
            for f in os.listdir(self.directory):
                if f.lower().endswith(supported_formats):
                    sources.append(os.path.join(self.directory, f))
        return sources

# ==========================================
# 2. Feature Conversion Module / 特徵轉換模組 (Feature Extractor)
# ==========================================
class FeatureExtractor:
    """
    Image Feature Extraction Class
    Uses a pre-trained MobileNetV3 model to convert images into high-dimensional feature vectors.
    影像特徵萃取類別
    使用輕量級 MobileNetV3 模型將影像轉換為高維度特徵向量，供 K-Means 分群使用
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"FeatureExtractor using device: {self.device} / 使用設備: {self.device}")
        
        # Load pre-trained MobileNetV3 Small model
        # 載入預訓練的 MobileNetV3 Small 模型
        self.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        # Remove the classification head, keep only the feature extraction part (Identity layer)
        # 移除最後的分類層，僅保留特徵萃取部分 (Identity 層)
        self.model.classifier = torch.nn.Identity()
        self.model.to(self.device)
        self.model.eval()
        
        # Define image preprocessing: resize to 224x224, normalization
        # 定義影像預處理流程：縮放至 224x224、標準化
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, image_paths):
        """
        Convert a list of images into feature vectors in batches to avoid OOM.
        將影像清單分批轉換為特徵向量，避免一次讀取太多影像導致記憶體崩潰
        """
        features = []
        batch_size = 32 # Batch size for processing
        
        with torch.no_grad():
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i+batch_size]
                batch_tensors = []
                for p in batch_paths:
                    img = cv2.imread(p)
                    if img is None:
                        continue
                    # BGR to RGB conversion (OpenCV uses BGR, PyTorch expects RGB)
                    # 顏色空間轉換 BGR -> RGB
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    tensor = self.transform(img_rgb)
                    batch_tensors.append(tensor)
                
                if not batch_tensors:
                    continue
                    
                # Stack batch tensors and perform inference
                # 合併批次張量並送入模型推論
                batch_tensor = torch.stack(batch_tensors).to(self.device)
                out = self.model(batch_tensor)
                features.append(out.cpu().numpy())
                
        if features:
            return np.vstack(features)
        return np.array([])

# ==========================================
# 3. Core Automated Pipeline / 核心自動化流水線 (Pipeline)
# ==========================================
class SamplingPipeline:
    """
    Core Pipeline Class
    Controls the entire "data acquisition -> inference filtering -> feature clustering -> weighted sampling" workflow.
    核心流水線類別
    控制整個「獲取資料 -> 推論過濾 -> 特徵分群 -> 加權抽樣」的流程
    """
    def __init__(self, model_path, output_dir):
        """
        Initialize Pipeline
        :param model_path: Path to YOLO model file (.engine or .pt)
        :param output_dir: Root directory for output
        """
        self.model_path = model_path
        self.output_dir = output_dir
        # Final output paths / 最終成果存放路徑
        self.auto_labeled_dir = os.path.join(self.output_dir, "auto_labeled")
        self.auto_images_dir = os.path.join(self.auto_labeled_dir, "images")
        self.auto_labels_dir = os.path.join(self.auto_labeled_dir, "labels")
        self.negative_sample_dir = os.path.join(self.output_dir, "negative_sample")
        
        # Disk temp directories to save candidates and avoid OOM
        # 磁碟暫存目錄，用於存放分群前的所有候選照片，以節省記憶體
        self.temp_negative_dir = os.path.join(self.output_dir, "temp_negative_candidates")
        self.temp_auto_dir = os.path.join(self.output_dir, "temp_auto_candidates")
        
        self.setup_directories()
        
        print(f"Loading YOLO model: {self.model_path}... / 正在載入 YOLO 模型...")
        self.yolo_model = YOLO(self.model_path)
        self.feature_extractor = FeatureExtractor()

    def setup_directories(self):
        """Ensure necessary directory structure exists. / 建立必要的目錄結構"""
        os.makedirs(self.auto_images_dir, exist_ok=True)
        os.makedirs(self.auto_labels_dir, exist_ok=True)
        os.makedirs(self.negative_sample_dir, exist_ok=True)
        os.makedirs(self.temp_negative_dir, exist_ok=True)
        os.makedirs(self.temp_auto_dir, exist_ok=True)

    def process_frame_results(self, results, frame, base_filename, negative_candidates, auto_candidates, frame_count, last_negative_frame, last_auto_frame):
        """
        Process single frame inference results, categorize by confidence, and save to temp areas.
        處理單幀的推論結果，根據信心度分類並決定是否存入暫存區
        """
        saved_count = 0
        for r in results:
            # Skip if no objects detected / 如果畫面上沒有偵測到任何物件則略過
            if len(r.boxes) == 0:
                continue
                
            confs = r.boxes.conf.cpu().numpy()
            if len(confs) == 0:
                continue
                
            # Get max detection confidence in frame / 獲取該幀中最高的偵測信心度
            max_conf = float(max(confs))
            
            # --- Category 1: Auto Labeled (High Confidence >= 0.8) ---
            if max_conf >= 0.8:
                # Force interval of at least 3 seconds (15 frames) to ensure diversity
                # 為了避免畫面過於重複，強制間隔至少 3 秒 (15 個處理幀)
                if frame_count - last_auto_frame >= 15:
                    # Save temp image / 暫存照片
                    img_path = os.path.join(self.temp_auto_dir, f"{base_filename}.jpg")
                    cv2.imwrite(img_path, frame)
                    
                    # Write label file (.txt), keeping only boxes with conf > 0.8
                    # 寫入標註檔 (.txt)，內容僅保留信心度 > 0.8 的物件框
                    txt_path = os.path.join(self.temp_auto_dir, f"{base_filename}.txt")
                    boxes_to_save = []
                    xywhn = r.boxes.xywhn.cpu().numpy()
                    cls_indices = r.boxes.cls.cpu().numpy()
                    
                    for i in range(len(confs)):
                        if confs[i] >= 0.8: # Strict filtering of boxes / 嚴格過濾單一框
                            c = int(cls_indices[i])
                            x, y, w, h = xywhn[i]
                            boxes_to_save.append(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                            
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        if boxes_to_save:
                            f.write('\n'.join(boxes_to_save))
                    
                    # Add to candidates / 加入候選清單
                    auto_candidates.append({
                        'path': img_path,
                        'txt_path': txt_path,
                        'max_conf': max_conf,
                        'name': base_filename
                    })
                    last_auto_frame = frame_count
                    saved_count += 1
                
            # --- Category 2: Negative Sample (Hard Negative Range 0.2 ~ 0.65) ---
            elif 0.2 < max_conf < 0.65:
                # 3-second interval rule for diversity / 加入 3 秒時間間隔限制
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
        """
        Iterate through dataset, perform inference extraction, and finally global sampling.
        遍歷整個資料集，執行推論擷取，最後統一進行全局抽樣
        """
        sources = dataset.get_sources()
        print(f"Found {len(sources)} sources to process. / 找到 {len(sources)} 個來源待處理。")
        
        negative_candidates = [] # Global negative candidates / 存放全局負樣本候選
        auto_candidates = []     # Global auto labeled candidates / 存放全局自動標記候選
        
        for source_idx, source in enumerate(sources):
            print(f"\nProcessing source {source_idx + 1}/{len(sources)}: {source}")
            
            is_youtube = type(source) is str and ("youtu" in source or "http" in source)
            is_image = type(source) is str and source.lower().endswith(('.jpg', '.jpeg', '.png'))
            
            # Handle single image / 處理單張圖片情況
            if is_image:
                print("Single image detected... / 偵測為單張影像處理中...")
                frame = cv2.imread(source)
                if frame is None:
                    continue
                results = self.yolo_model.predict(source=frame, conf=0.2, verbose=False)
                file_basename = os.path.splitext(os.path.basename(source))[0]
                base_filename = f"{file_basename}_img"
                saved, _, _ = self.process_frame_results(results, frame, base_filename, negative_candidates, auto_candidates, 15, 0, 0)
                print(f"Image completed. Collected {saved} auto_labeled candidates.")
                continue

            # Process video or YouTube stream (using OpenCV frame extraction)
            # 處理影片或 YouTube 串流 (利用 OpenCV 抽幀)
            video_path = source
            if is_youtube:
                print("Extracting stream URL using yt-dlp... / 正在使用 yt-dlp 解析串流網址...")
                try:
                    # yt-dlp avoids "Waiting for stream 0" and JS runtime issues.
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
                print(f"Failed to open source: {source}")
                continue
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or np.isnan(fps):
                fps = 30.0 # Default 30 fps
                
            # Stride for extracting 5 frames per second (5 FPS)
            # 計算每秒抽取 5 幀 (5 fps) 的間隔
            frame_stride = max(1, int(round(fps / 5.0)))
            print(f"Video FPS: {fps:.2f}, Stride: {frame_stride} (approx 5 FPS)")
            
            frame_count = 0     # Actual frame index
            processed_count = 0 # Count of processed frames after stride
            saved_count = 0     # Final count saved to temp
            last_negative_frame = -999
            last_auto_frame = -999
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Extract frames based on calculated stride / 按照 stride 抽幀
                if frame_count % frame_stride == 0:
                    # Perform YOLO inference / 將影像送入 YOLO
                    results = self.yolo_model.predict(source=frame, conf=0.2, verbose=False)
                    
                    if is_youtube:
                        base_filename = f"youtube_{source_idx}_frame_{frame_count}"
                    else:
                        file_basename = os.path.splitext(os.path.basename(source))[0]
                        base_filename = f"{file_basename}_frame_{frame_count}"
                    
                    # Process inference results / 處理推論結果
                    saved, last_negative_frame, last_auto_frame = self.process_frame_results(
                        results, frame, base_filename, negative_candidates, auto_candidates, processed_count, last_negative_frame, last_auto_frame
                    )
                    saved_count += saved
                    processed_count += 1
                    
                frame_count += 1
            
            cap.release()
            print(f"Source completed. Collected {saved_count} candidates.")
            
        print(f"\n--- Global Collection Completed / 全局收集完成 ---")
        print(f"Total Negative Candidates: {len(negative_candidates)}")
        print(f"Total Auto Candidates: {len(auto_candidates)}")
        
        # 1. Execute global negative clustering and weighted sampling
        # 執行全局負樣本分群與加權抽樣
        if len(negative_candidates) > 0:
            self.cluster_and_sample_negative(negative_candidates)
        else:
            print("No suitable negative candidates found.")
            
        # 2. Execute global auto clustering and Top-10 selection (call external module)
        # 執行全局自動標記分群與 Top 10 選取
        if len(auto_candidates) > 0:
            # Instantiate AutoLabelClassifier and execute selection
            auto_classifier = AutoLabelClassifier(self.feature_extractor, self.auto_images_dir, self.auto_labels_dir)
            auto_classifier.cluster_and_select(auto_candidates, top_k=10)
        else:
            print("No suitable auto labeled candidates found.")
            
        # Clean up temp directories / 最後清理磁碟暫存目錄
        if os.path.exists(self.temp_negative_dir):
            shutil.rmtree(self.temp_negative_dir)
        if os.path.exists(self.temp_auto_dir):
            shutil.rmtree(self.temp_auto_dir)
            
        print("\n✅ Pipeline execution completed! Results saved to auto_labeled and negative_sample.")


    def cluster_and_sample_negative(self, candidates):
        """
        Implements K-Means clustering, deception metric, Z-score normalization, and weighted random sampling (Negative Only).
        實作 K-Means 分群、定義難度指標、Z-score 正規化與隨機加權抽樣 (負樣本專用)
        """
        print("\n--- Processing Negative Samples / 正在處理負樣本 ---")
        print("Extracting features... / 正在擷取候選照片特徵...")
        image_paths = [c['path'] for c in candidates]
        features = self.feature_extractor.extract_features(image_paths)
        
        print("Performing K-Means clustering... / 執行 K-Means 全局分群中...")
        # Cluster all candidates into 10 groups (or fewer if insufficient data)
        # 分為 10 群
        n_clusters = min(10, len(candidates)) 
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features)
        
        # Calculate confidence distribution by cluster
        # 依群組計算信心度分布
        cluster_confs = {i: [] for i in range(n_clusters)}
        for idx, cluster_id in enumerate(clusters):
            cluster_confs[cluster_id].append(candidates[idx]['max_conf'])
            
        # Deception Metric: mean of max confidence per cluster (higher = more deceptive)
        # 定義難度指標：每一群內的平均最高信心度
        cluster_metrics = {}
        for i in range(n_clusters):
            if cluster_confs[i]:
                cluster_metrics[i] = np.mean(cluster_confs[i])
            else:
                cluster_metrics[i] = 0.0
                
        metrics_list = [cluster_metrics[i] for i in range(n_clusters)]
        print(f"Cluster metrics (Mean Conf): {metrics_list}")
        
        # Convert metrics to probability weights w using Z-score
        # 使用 Z-score 將難度轉化為抽樣機率權重 w
        if n_clusters > 1 and np.std(metrics_list) > 0:
            z_scores = zscore(metrics_list)
            # Softmax mapping of Z-score to (0, 1) distribution
            exp_z = np.exp(z_scores)
            weights = exp_z / np.sum(exp_z)
        else:
            weights = np.ones(n_clusters) / n_clusters
            
        print(f"Sampling weights distribution w: {weights}")
        
        # Determine final sample count (30% of candidates, cap at 1000)
        # 決定最終抽樣總數
        n_samples = min(1000, max(1, int(len(candidates) * 0.3)))
        
        # Assign cluster weights to individual images
        # 將群組權重分配給個別照片
        cand_probs = []
        for idx, cluster_id in enumerate(clusters):
            cluster_size = len(cluster_confs[cluster_id])
            # Prob per image = group_weight / group_size
            prob = weights[cluster_id] / cluster_size
            cand_probs.append(prob)
            
        cand_probs = np.array(cand_probs)
        cand_probs /= cand_probs.sum() # Normalize sum to 1
        
        # Weighted random choice without replacement / 隨機加權抽樣
        sampled_indices = np.random.choice(
            len(candidates), 
            size=n_samples, 
            replace=False, 
            p=cand_probs
        )
        
        print(f"Saving {n_samples} final negative samples... / 正在儲存最終負樣本...")
        for idx in sampled_indices:
            cand = candidates[idx]
            dest_path = os.path.join(self.negative_sample_dir, f"{cand['name']}.jpg")
            if os.path.exists(cand['path']):
                shutil.copy(cand['path'], dest_path) # Move from temp to final


# ==========================================
# 4. Main Entry / 程式進入點
# ==========================================
def main():
    """Main menu interface. / 主選單介面"""
    print("=== Traffic Violation Reporting Automation (TVRA) ===")
    print("Data Source Mode / 請選擇資料來源模式:")
    print("1. YouTube Stream (Automated Scraping) / YouTube 影片串流 (自動爬取)")
    print("2. Local Folder / 本地影片或照片資料夾")
    choice = input("Enter choice (1 or 2): / 請輸入選項 (1 或 2): ").strip()
    
    # Create appropriate Dataset instance
    if choice == '1':
        num = input("Number of YouTube videos to process (default 200): / 請輸入欲處理的 YouTube 影片數量 (預設 200): ").strip()
        num = int(num) if num.isdigit() else 200
        dataset = YoutubeDataset(target_count=num)
    elif choice == '2':
        # Default to parent dir's test_video as script runs in negative_clean/
        default_dir = os.path.join("..", "test_video")
        local_dir = input(f"Input folder path (default: {default_dir}): / 請輸入本地資料夾路徑 (預設: {default_dir}): ").strip()
        if not local_dir:
            local_dir = default_dir
            if not os.path.exists(local_dir):
                os.makedirs(local_dir, exist_ok=True)
                print(f"Created {local_dir} folder. Please add files and restart. / 已自動創建資料夾，請放入檔案後重新執行。")
                return
        dataset = LocalDataset(local_dir)
    else:
        print("Invalid choice. / 無效選擇，結束程式。")
        return

    # Path to YOLO model (prefer TensorRT engine, fallback to pt)
    # 指定 YOLO 模型路徑
    engine_path = os.path.join("..", "YOLO_V4_Result", "train", "weights", "best.engine")
    pt_path = os.path.join("..", "YOLO_V4_Result", "train", "weights", "best.pt")
    
    if os.path.exists(engine_path):
        model_path = engine_path
    elif os.path.exists(pt_path):
        print(f"Fallback to PyTorch model: {pt_path} / 使用 PyTorch 模型")
        model_path = pt_path
    else:
        print(f"Default model path not found. / 找不到預設的模型路徑。")
        model_path = input("Enter manual path (e.g. ../yolo26n.pt): ").strip()
        if not model_path:
            return

    # Initialize and run Pipeline
    # 因腳本位於 negative_clean 資料夾，將輸出目錄設為當前目錄 (.)
    pipeline = SamplingPipeline(model_path=model_path, output_dir=".")
    pipeline.process_dataset(dataset)

if __name__ == "__main__":
    main()
