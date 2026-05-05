import shutil
import argparse
import logging
from pathlib import Path
from embedding import ImageDeduplicator

logger = logging.getLogger(__name__)

class DeduplicationService:
    """去重流程服務類別"""
    def __init__(self, threshold: float, yolo_weights: str = None):
        self._deduplicator = ImageDeduplicator(threshold=threshold, yolo_weights=yolo_weights)
        
    def execute(self, input_folder: Path, output_folder: Path, use_confidence: bool) -> None:
        if not input_folder.exists() or not input_folder.is_dir():
            logger.error(f"輸入資料夾不存在或無效: {input_folder}")
            return

        # 讀取並替換路徑斜線
        file_list = [str(f.resolve()).replace("\\", "/") for f in input_folder.iterdir() if f.is_file()]
        
        if not file_list:
            logger.warning("輸入資料夾內沒有檔案")
            return

        # 執行去重邏輯
        logger.info("開始執行圖片去重分析...")
        final_files = self._deduplicator.process_batch(
            file_list, 
            use_confidence=use_confidence, 
            sample_way="negative"
        )

        logger.info(f"原始圖片數量: {len(file_list)}")
        logger.info(f"去重後保留數量: {len(final_files)}")
        logger.info("-" * 30)

        # 建立新資料夾並寫入圖片
        output_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"準備將圖片複製到: {output_folder}")

        for file_path in final_files:
            file_name = Path(file_path).name 
            dest_path = output_folder / file_name 
            shutil.copy(file_path, dest_path)

        logger.info("所有乾淨圖片已成功寫入新資料夾！")

def main():
    parser = argparse.ArgumentParser(description="圖片去重分析工具")
    parser.add_argument("--input_folder", type=str, required=True, help="原始圖片資料夾路徑")
    parser.add_argument("--output_folder", type=str, required=True, help="去重後保留圖片的輸出資料夾路徑")
    parser.add_argument("--threshold", type=float, default=0.90, help="相似度閥值，預設為 0.90")
    parser.add_argument("--yolo_weights", type=str, default=None, help="YOLO 權重檔案路徑 (若不需 YOLO 信心度可省略)")
    parser.add_argument("--use_confidence", action="store_true", help="是否啟用 YOLO 信心度進行保留策略")
    
    args = parser.parse_args()

    # 設定全域 Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    service = DeduplicationService(threshold=args.threshold, yolo_weights=args.yolo_weights)
    service.execute(
        input_folder=Path(args.input_folder),
        output_folder=Path(args.output_folder),
        use_confidence=args.use_confidence
    )

if __name__ == "__main__":
    main()
