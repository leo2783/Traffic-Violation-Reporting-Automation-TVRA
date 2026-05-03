import shutil
from pathlib import Path
from embedding import ImageDeduplicator

# 1. 設定原始資料夾與目標資料夾
folder = Path(r"C:/Users/qet63/Documents/Traffic-Violation-Reporting-Automation-TVRA-/negative_sample")
# 新增一個用來存放乾淨圖片的資料夾路徑
output_folder = Path(r"C:/Users/qet63/Documents/Traffic-Violation-Reporting-Automation-TVRA-/cleaned_images1")

# 讀取並替換路徑斜線
file_list = [str(f.resolve()).replace("\\", "/") for f in folder.iterdir() if f.is_file()]

# 2. 執行去重邏輯，取得保留清單
print("開始執行圖片去重分析...")
final_files = ImageDeduplicator(threshold=0.90).process_batch(file_list, confident=True, sample_way="negative")

print(f"原始圖片數量: {len(file_list)}")
print(f"去重後保留數量: {len(final_files)}")
print("-" * 30)

# 3. 建立新資料夾並寫入圖片
# exist_ok=True 代表如果資料夾已經存在，程式不會報錯
output_folder.mkdir(parents=True, exist_ok=True)

print(f"準備將圖片複製到: {output_folder}")

for file_path in final_files:
    # 取出檔名 (例如: '00000.jpg')
    file_name = Path(file_path).name 
    
    # 組合成目標路徑 (例如: '.../cleaned_images/00000.jpg')
    dest_path = output_folder / file_name 
    
    # 執行複製動作
    shutil.copy(file_path, dest_path)

print("所有乾淨圖片已成功寫入新資料夾！")