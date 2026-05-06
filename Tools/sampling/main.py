import argparse
import logging
from pathlib import Path
from typing import Optional, Callable

try:
    from .services import DeduplicationService
except ImportError:
    from services import DeduplicationService

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="圖片去重分析工具")
    parser.add_argument("--input_folder", type=str, required=True, help="原始圖片資料夾路徑")
    parser.add_argument("--output_folder", type=str, required=True, help="去重後保留圖片的輸出資料夾路徑")
    parser.add_argument("--threshold", type=float, default=0.90, help="相似度閥值，預設為 0.90")
    parser.add_argument("--yolo_weights", type=str, default=None, help="YOLO 權重檔案路徑 (若不需 YOLO 信心度可省略)")
    parser.add_argument("--use_confidence", action="store_true", help="是否啟用 YOLO 信心度進行保留策略")
    parser.add_argument("--sample_way", type=str, choices=["negative", "positive"], default="negative", help="YOLO 信心度排序策略")
    parser.add_argument("--write_mode", type=str, choices=["per-folder", "per-video", "per-frame"], default="per-folder", help="寫入模式")
    
    args = parser.parse_args()

    # 設定全域 Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    service = DeduplicationService(threshold=args.threshold, yolo_weights=args.yolo_weights)
    service.execute(
        input_folder=Path(args.input_folder),
        output_folder=Path(args.output_folder),
        use_confidence=args.use_confidence,
        sample_way=args.sample_way,
        write_mode=args.write_mode
    )

if __name__ == "__main__":
    main()
