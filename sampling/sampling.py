from extract_negative import NegativeSampler
from pathlib import Path

folder = Path(r"C:/Users/qet63/Documents/Traffic-Violation-Reporting-Automation-TVRA-/cleaned_images1")
# 新增一個用來存放乾淨圖片的資料夾路徑
output_folder = Path(r"C:/Users/qet63/Documents/Traffic-Violation-Reporting-Automation-TVRA-/cleaned_images2")

# 讀取並替換路徑斜線
file_list = [str(f.resolve()).replace("\\", "/") for f in folder.iterdir() if f.is_file()]
final=NegativeSampler().sample(image_paths=file_list,num_samples=423)