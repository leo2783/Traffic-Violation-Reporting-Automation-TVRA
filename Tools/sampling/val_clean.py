import shutil
import argparse
import logging
from pathlib import Path
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class ValidationCleaner:
    """YOLO 驗證集清洗工具類別"""
    def __init__(self, yolo_weights: str, threshold: float = 0.8):
        if not isinstance(threshold, float) or not (0.0 <= threshold <= 1.0):
            raise ValueError("threshold 必須是 0.0 到 1.0 之間的浮點數")
        self._yolo_weights = yolo_weights
        self._threshold = threshold
        self._model = YOLO(yolo_weights, task="detect")
        
    def infer_and_filter(self, source_path: str, out_path: str) -> None:
        source_dir = Path(source_path)
        out_dir = Path(out_path)
        
        if not source_dir.exists() or not source_dir.is_dir():
            logger.error(f"來源資料夾不存在或無效: {source_dir}")
            return
            
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
                    results = self._model.predict(str(img_file), verbose=False)
                    result = results[0] 
                    
                    # 檢查是否有偵測到物件
                    if len(result.boxes) > 0:
                        # 取得該張圖片中最高的信心度分數
                        max_conf = result.boxes.conf.max().item()
                        
                        if max_conf >= self._threshold:
                            logger.info(f"[保留] {img_file.name} (最高信心度: {max_conf:.2f})")
                            
                            # 1. 複製圖片
                            shutil.copy(img_file, out_images / img_file.name)
                            
                            # 2. 產生並儲存標註檔 (含信心度)
                            txt_path = out_labels / f"{img_file.stem}.txt"
                            result.save_txt(str(txt_path), save_conf=True)
                except Exception as e:
                    logger.warning(f"處理圖片失敗 {img_file.name}: {e}")
                    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(yolo_weights='{self._yolo_weights}', threshold={self._threshold})>"

def main():
    parser = argparse.ArgumentParser(description="YOLO 驗證集清洗工具")
    parser.add_argument("--yolo_weights", type=str, required=True, help="YOLO 權重檔案路徑")
    parser.add_argument("--source_path", type=str, required=True, help="來源圖片資料夾路徑")
    parser.add_argument("--out_path", type=str, required=True, help="清洗後輸出的資料夾路徑")
    parser.add_argument("--threshold", type=float, default=0.6, help="信心度門檻 (預設 0.6)")
    
    args = parser.parse_args()
    
    # 確保入口檔案有設定 logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    cleaner = ValidationCleaner(yolo_weights=args.yolo_weights, threshold=args.threshold)
    cleaner.infer_and_filter(source_path=args.source_path, out_path=args.out_path)

if __name__ == "__main__":
    main()
