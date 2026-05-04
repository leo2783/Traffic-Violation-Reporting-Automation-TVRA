import os
from shapely.geometry import Polygon

def check_yolo_seg_labels(label_dir, min_area=1e-5):
    issues = []
    
    # 檢查目錄是否存在
    if not os.path.exists(label_dir):
        print(f"找不到目錄: {label_dir}")
        return issues

    for file in os.listdir(label_dir):
        if not file.endswith('.txt'):
            continue
            
        filepath = os.path.join(label_dir, file)
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        for line_idx, line in enumerate(lines):
            parts = line.strip().split()
            # YOLO seg 格式: class_id x1 y1 x2 y2 ... (至少要有 3 個點，即 1+6=7 個數值)
            if len(parts) < 7: 
                issues.append((file, line_idx + 1, "頂點過少 (<3個點)"))
                continue
                
            try:
                coords = [float(x) for x in parts[1:]]
            except ValueError:
                issues.append((file, line_idx + 1, "包含非數值字元"))
                continue
            
            # 1. 檢查座標是否越界 (正常歸一化座標應介於 0~1)
            if any(c < 0.0 or c > 1.0 for c in coords):
                issues.append((file, line_idx + 1, "座標越界 (<0 或 >1)"))
                continue
                
            # 2. 構建多邊形
            points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
            poly = Polygon(points)
            
            # 3. 檢查自交或無效幾何結構
            if not poly.is_valid:
                issues.append((file, line_idx + 1, "多邊形無效 (通常是線段自交)"))
                continue
                
            # 4. 檢查面積是否極小 (過小的標註在透視變換時極易除以零)
            if poly.area < min_area:
                issues.append((file, line_idx + 1, f"面積極小 ({poly.area:.1e})"))
                continue
                
    return issues

if __name__ == "__main__":
    # 將這裡替換為您的標註檔資料夾路徑
    label_folder = r"C:/Users/qet63/Documents/Traffic-Violation-Reporting-Automation-TVRA-/YOLO_V5_datasets/test2/poly_dataset/labels/val"
    
    print(f"開始掃描 {label_folder} ...\n")
    bad_labels = check_yolo_seg_labels(label_folder)

    if not bad_labels:
        print("✅ 掃描完成！沒有發現明顯的瑕疵標註。")
    else:
        print(f"❌ 發現 {len(bad_labels)} 個問題標註：")
        for filename, line_num, reason in bad_labels:
            print(f"檔案: {filename} | 行號: {line_num} | 問題: {reason}")