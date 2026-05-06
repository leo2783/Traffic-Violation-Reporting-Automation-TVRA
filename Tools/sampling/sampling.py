import argparse
import logging
try:
    from .extract_negative import NegativeSampler
except ImportError:
    from extract_negative import NegativeSampler
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="負樣本抽樣工具")
    parser.add_argument("--input_folder", type=str, required=True, help="輸入去重後的圖片資料夾路徑")
    parser.add_argument("--output_folder", type=str, required=True, help="抽樣結果存放資料夾路徑")
    parser.add_argument("--num_samples", type=int, required=True, help="要抽樣的圖片數量")
    parser.add_argument("--yolo_weights", type=str, required=True, help="YOLO 權重檔案路徑")
    parser.add_argument("--temperature", type=float, default=5.0, help="抽樣機率分布的 Temperature 參數")
    
    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    output_folder = args.output_folder

    if not input_folder.exists():
        logging.error(f"輸入資料夾不存在: {input_folder}")
        return

    # 讀取並替換路徑斜線
    file_list = [str(f.resolve()).replace("\\", "/") for f in input_folder.iterdir() if f.is_file()]
    
    if not file_list:
        logging.warning("輸入資料夾內沒有檔案")
        return

    sampler = NegativeSampler(yolo_weights=args.yolo_weights, temperature=args.temperature)
    final_paths = sampler.sample(
        image_paths=file_list, 
        num_samples=args.num_samples, 
        output_dir=output_folder
    )
    
    logging.info(f"抽樣完成，共獲得 {len(final_paths)} 張圖片。")

if __name__ == "__main__":
    main()
