import os
import sys
import cv2
import shutil
# 加入 negative_clean 目錄到環境變數以支援 import
sys.path.insert(0, os.path.abspath('negative_clean'))

from pipeline import FeatureExtractor
from auto_label_classifier import AutoLabelClassifier
from ultralytics import YOLO

def main():
    print("=== Auto Labeled Folder Classifier ===")
    input_dir = r"C:/Users/qet63/Documents/Traffic-Violation-Reporting-Automation-TVRA-/auto_labeled"
    images_dir = os.path.join(input_dir, "images")
    labels_dir = os.path.join(input_dir, "labels")
    
    # 將結果輸出到一個新的獨立資料夾，避免覆蓋原始資料
    output_dir = r"C:/Users/qet63/Documents/Traffic-Violation-Reporting-Automation-TVRA-/auto_labeled_top20"
    out_images_dir = os.path.join(output_dir, "images")
    out_labels_dir = os.path.join(output_dir, "labels")
    
    if not os.path.exists(images_dir):
        print(f"錯誤：找不到輸入的圖片資料夾 {images_dir}")
        return

    # 確保輸出目錄存在且乾淨 (可根據需求選擇是否先清空)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)

    # 載入 YOLO 模型
    engine_path = os.path.join("YOLO_V4_Result", "train", "weights", "best.engine")
    pt_path = os.path.join("YOLO_V4_Result", "train", "weights", "best.pt")
    
    if os.path.exists(engine_path):
        model_path = engine_path
    elif os.path.exists(pt_path):
        model_path = pt_path
    else:
        print(f"找不到模型檔案: {engine_path} 或 {pt_path}")
        return
        
    print(f"Loading YOLO model from {model_path}...")
    yolo_model = YOLO(model_path)
    
    candidates = []
    supported_formats = ('.jpg', '.jpeg', '.png')
    
    print(f"Scanning images in {images_dir}...")
    # 掃描資料夾內的所有圖片
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(supported_formats)]
    
    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(images_dir, img_file)
        base_name = os.path.splitext(img_file)[0]
        txt_path = os.path.join(labels_dir, f"{base_name}.txt")
        
        # 重新用 YOLO 推論以取得信心度 max_conf
        results = yolo_model.predict(source=img_path, conf=0.2, verbose=False)
        max_conf = 0.0
        
        for r in results:
            if len(r.boxes) > 0:
                confs = r.boxes.conf.cpu().numpy()
                if len(confs) > 0:
                    current_max = float(max(confs))
                    if current_max > max_conf:
                        max_conf = current_max
        
        # 將讀取到的資訊包裝成 dict 加入候選清單
        candidates.append({
            'path': img_path,
            'txt_path': txt_path if os.path.exists(txt_path) else None,
            'max_conf': max_conf,
            'name': base_name
        })
        
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(image_files)} images...")
            
    print(f"\nTotal candidates prepared: {len(candidates)}")
    
    if len(candidates) > 0:
        print("\nInitializing FeatureExtractor and Classifier...")
        feature_extractor = FeatureExtractor()
        classifier = AutoLabelClassifier(feature_extractor, out_images_dir, out_labels_dir)
        
        # 開始分群並挑選前 10 名
        classifier.cluster_and_select(candidates, top_k=5)
        
        print(f"\n✅ 完成！已將分群篩選後的前 10 名檔案存入: {output_dir}")
    else:
        print("沒有找到任何候選檔案可供處理。")

if __name__ == "__main__":
    main()
