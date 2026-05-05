import shutil
import argparse
import logging
from pathlib import Path
from typing import List, Optional

from utils import YoloAnalyzer, path_check

logger = logging.getLogger(__name__)

class ValidationCleaner:
    """
    YOLO 驗證集清洗工具類別
    負責根據模型推論的信心度門檻過濾圖片，並同步產生對應的標註檔。
    """
    def __init__(self, yolo_weights: str, threshold: float = 0.6):
        """
        初始化清洗工具
        :param yolo_weights: YOLO 模型權重路徑
        :param threshold: 信心度門檻 (0.0 ~ 1.0)
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("threshold 必須在 0.0 到 1.0 之間")
            
        self._threshold = threshold
        # 使用專案標準的 YoloAnalyzer
        self._analyzer = YoloAnalyzer(yolo_weights)
        
    @property
    def threshold(self) -> float:
        return self._threshold
        
    @threshold.setter
    def threshold(self, value: float):
        if not (0.0 <= value <= 1.0):
            raise ValueError("threshold 必須在 0.0 到 1.0 之間")
        self._threshold = value

    def clean(self, source_path: str, out_path: str) -> None:
        """
        執行清洗流程
        :param source_path: 來源圖片資料夾路徑
        :param out_path: 輸出結果資料夾路徑
        """
        source_dir = Path(source_path)
        out_dir = Path(out_path)
        
        if not source_dir.exists() or not source_dir.is_dir():
            logger.error(f"來源資料夾不存在或無效: {source_dir}")
            return
            
        # 建立輸出目錄結構
        out_images = out_dir / "images"
        out_labels = out_dir / "labels"
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        # 收集有效圖片路徑
        img_files = [f for f in source_dir.iterdir() if path_check(str(f))]
        
        if not img_files:
            logger.warning(f"在 {source_dir} 中未找到有效的圖片檔案")
            return

        logger.info(f"開始分析 {len(img_files)} 張圖片 (門檻: {self._threshold})...")
        
        # 執行批次推論 (YoloAnalyzer 已內建 stream=True 與 GPU 支援)
        img_paths = [str(f) for f in img_files]
        results = self._analyzer.predict(img_paths, verbose=False)
        
        count = 0
        for result in results:
            try:
                img_path_str = result.path
                if not img_path_str:
                    continue
                    
                img_file = Path(img_path_str)
                
                # 檢查是否有物件且信心度達標
                if len(result.boxes) > 0:
                    max_conf = result.boxes.conf.max().item()
                    
                    if max_conf >= self._threshold:
                        # 複製圖片
                        shutil.copy(img_file, out_images / img_file.name)
                        
                        # 儲存標註檔
                        txt_path = out_labels / f"{img_file.stem}.txt"
                        result.save_txt(str(txt_path), save_conf=True)
                        
                        logger.debug(f"[保留] {img_file.name} ({max_conf:.2f})")
                        count += 1
            except Exception as e:
                logger.warning(f"處理推論結果失敗: {e}")

        logger.info(f"清洗完成！共保留 {count} 張圖片與標註檔，存放在: {out_dir}")

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(threshold={self._threshold})>"

def main():
    parser = argparse.ArgumentParser(description="YOLO 驗證集清洗工具 (Refactored)")
    parser.add_argument("--yolo_weights", type=str, required=True, help="YOLO 權重檔案路徑")
    parser.add_argument("--source_path", type=str, required=True, help="來源圖片資料夾路徑")
    parser.add_argument("--out_path", type=str, required=True, help="清洗後輸出的資料夾路徑")
    parser.add_argument("--threshold", type=float, default=0.6, help="信心度門檻 (預設 0.6)")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    cleaner = ValidationCleaner(yolo_weights=args.yolo_weights, threshold=args.threshold)
    cleaner.clean(source_path=args.source_path, out_path=args.out_path)

if __name__ == "__main__":
    main()
