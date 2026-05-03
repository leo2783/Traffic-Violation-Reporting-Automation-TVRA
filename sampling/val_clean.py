import shutil
from pathlib import Path
from ultralytics import YOLO

def infer_and_filter(threshold=0.8):
    model = YOLO(r"C:/Users/qet63/Documents/Traffic-Violation-Reporting-Automation-TVRA-/YOLO_V4_Result/train/weights/best.pt",
                             task="detect"
                             )
    
    source_path = Path(r"C:/Users/qet63/Pictures/n/images")
    out_path = Path(r"C:/Users/qet63/Documents/Traffic-Violation-Reporting-Automation-TVRA-/val_cleaned")
    
    # 在輸出目錄下分別建立 images 和 labels 資料夾
    out_images = out_path / "images"
    out_labels = out_path / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    for img_file in source_path.iterdir():
        if img_file.suffix.lower() in img_extensions:
            # 針對單張圖片執行推論 
            results = model(img_file, verbose=False)
            result = results[0] 
            
            # 檢查是否有偵測到物件
            if len(result.boxes) > 0:
                # 取得該張圖片中最高的信心度分數
                max_conf = result.boxes.conf.max().item()
                
                if max_conf >= threshold:
                    print(f"[保留] {img_file.name} (最高信心度: {max_conf:.2f})")
                    
                    # 1. 複製圖片
                    shutil.copy(img_file, out_images / img_file.name)
                    
                    # 2. 產生並儲存標註檔 (含信心度)
                    txt_path = out_labels / f"{img_file.stem}.txt"
                    result.save_txt(txt_path, save_conf=True)


infer_and_filter(
    threshold=0.6                  # 信心度門檻
)