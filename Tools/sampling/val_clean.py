import shutil
import argparse
import logging
from pathlib import Path
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def infer_and_filter(yolo_weights, source_path, out_path, threshold=0.8):
    source_dir = Path(source_path)
    out_dir = Path(out_path)
    
    if not source_dir.exists():
        logging.error(f"來源資料夾不存在: {source_dir}")
        return

    model = YOLO(yolo_weights, task="detect")
    
    # 在輸出目錄下分別建立 images 和 labels 資料夾
    out_images = out_dir / "images"
    out_labels = out_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    for img_file in source_dir.iterdir():
        if img_file.suffix.lower() in img_extensions:
            try:
                # 針對單張圖片執行推論 
                results = model.predict(str(img_file), verbose=False)
                result = results[0] 
                
                # 檢查是否有偵測到物件
                if len(result.boxes) > 0:
                    # 取得該張圖片中最高的信心度分數
                    max_conf = result.boxes.conf.max().item()
                    
                    if max_conf >= threshold:
                        logging.info(f"[保留] {img_file.name} (最高信心度: {max_conf:.2f})")
                        
                        # 1. 複製圖片
                        shutil.copy(img_file, out_images / img_file.name)
                        
                        # 2. 產生並儲存標註檔 (含信心度)
                        txt_path = out_labels / f"{img_file.stem}.txt"
                        result.save_txt(str(txt_path), save_conf=True)
            except Exception as e:
                logging.warning(f"處理圖片失敗 {img_file.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="YOLO 驗證集清洗工具")
    parser.add_argument("--yolo_weights", type=str, required=True, help="YOLO 權重檔案路徑")
    parser.add_argument("--source_path", type=str, required=True, help="來源圖片資料夾路徑")
    parser.add_argument("--out_path", type=str, required=True, help="清洗後輸出的資料夾路徑")
    parser.add_argument("--threshold", type=float, default=0.6, help="信心度門檻 (預設 0.6)")
    
    args = parser.parse_args()
    
    infer_and_filter(
        yolo_weights=args.yolo_weights,
        source_path=args.source_path,
        out_path=args.out_path,
        threshold=args.threshold
    )

if __name__ == "__main__":
    main()
