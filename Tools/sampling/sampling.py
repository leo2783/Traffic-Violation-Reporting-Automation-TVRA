import argparse
import logging
from pathlib import Path

try:
    from .services import NegativeSamplingService
except ImportError:
    from services import NegativeSamplingService

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="負樣本抽樣工具")
    parser.add_argument("--input_folder", type=str, required=True, help="輸入去重後的圖片資料夾路徑")
    parser.add_argument("--output_folder", type=str, required=True, help="抽樣結果存放資料夾路徑")
    parser.add_argument("--num_samples", type=int, required=True, help="要抽樣的圖片數量")
    parser.add_argument("--yolo_weights", type=str, required=True, help="YOLO 權重檔案路徑")
    parser.add_argument("--temperature", type=float, default=5.0, help="抽樣機率分布的 Temperature 參數")
    
    args = parser.parse_args()

    service = NegativeSamplingService(
        yolo_weights=args.yolo_weights,
        temperature=args.temperature,
    )
    final_paths = service.execute(
        input_folder=Path(args.input_folder),
        output_folder=Path(args.output_folder),
        num_samples=args.num_samples,
    )
    
    logging.info(f"抽樣完成，共獲得 {len(final_paths)} 張圖片。")

if __name__ == "__main__":
    main()
