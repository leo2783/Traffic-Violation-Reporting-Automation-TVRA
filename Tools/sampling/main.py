import shutil
import argparse
import logging
from pathlib import Path
from typing import Optional, Callable
try:
    from .embedding import ImageDeduplicator
except ImportError:
    from embedding import ImageDeduplicator

logger = logging.getLogger(__name__)

class DeduplicationService:
    """去重流程服務類別"""
    def __init__(self, threshold: float, yolo_weights: str = None):
        self._deduplicator = ImageDeduplicator(threshold=threshold, yolo_weights=yolo_weights)
        
    def execute(self, input_folder: Path, output_folder: Path, use_confidence: bool,
                write_mode: str = "per-folder", progress_callback: Optional[Callable[[int, int], None]] = None) -> None:
        if not input_folder.exists() or not input_folder.is_dir():
            logger.error(f"輸入資料夾不存在或無效: {input_folder}")
            return

        # 讀取並過濾出圖片檔案
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        file_list = [
            f.resolve().as_posix() 
            for f in input_folder.iterdir() 
            if f.is_file() and f.suffix.lower() in valid_extensions
        ]
        
        if not file_list:
            logger.warning("輸入資料夾內沒有有效的圖片檔案 (支援格式: jpg, jpeg, png, bmp, webp)")
            return

        # 執行去重邏輯
        logger.info("開始執行圖片去重分析...")
        
        # 建立輸出資料夾
        output_folder.mkdir(parents=True, exist_ok=True)
        
        final_files = self._deduplicator.process_batch(
            file_list, 
            use_confidence=use_confidence, 
            sample_way="negative",
            write_mode=write_mode,
            output_folder=str(output_folder),
            progress_callback=progress_callback
        )

        logger.info(f"原始圖片數量: {len(file_list)}")
        logger.info(f"去重後總計保留數量: {len(final_files)}")
        logger.info("-" * 30)
        logger.info("所有乾淨圖片已處理完畢！")

def main():
    parser = argparse.ArgumentParser(description="圖片去重分析工具")
    parser.add_argument("--input_folder", type=str, required=True, help="原始圖片資料夾路徑")
    parser.add_argument("--output_folder", type=str, required=True, help="去重後保留圖片的輸出資料夾路徑")
    parser.add_argument("--threshold", type=float, default=0.90, help="相似度閥值，預設為 0.90")
    parser.add_argument("--yolo_weights", type=str, default=None, help="YOLO 權重檔案路徑 (若不需 YOLO 信心度可省略)")
    parser.add_argument("--use_confidence", action="store_true", help="是否啟用 YOLO 信心度進行保留策略")
    parser.add_argument("--write_mode", type=str, choices=["per-folder", "per-video", "per-frame"], default="per-folder", help="寫入模式")
    
    args = parser.parse_args()

    # 設定全域 Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    service = DeduplicationService(threshold=args.threshold, yolo_weights=args.yolo_weights)
    service.execute(
        input_folder=Path(args.input_folder),
        output_folder=Path(args.output_folder),
        use_confidence=args.use_confidence,
        write_mode=args.write_mode
    )

if __name__ == "__main__":
    main()
