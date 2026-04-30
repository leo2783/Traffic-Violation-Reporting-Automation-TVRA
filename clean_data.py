import os
import shutil

def main():
    target_dir = os.path.join("to_check", "negative_sample")
    img_dir = os.path.join(target_dir, "images")
    lbl_dir = os.path.join(target_dir, "labels")
    
    # 確認路徑是否存在
    if not os.path.exists(img_dir) and not os.path.exists(lbl_dir):
        print("找不到 images 或 labels 資料夾！")
        return

    # 定義分類後的資料夾路徑
    no_label_dir = os.path.join(target_dir, "no_label")
    with_label_dir = os.path.join(target_dir, "with_label")
    
    no_label_img_dir = os.path.join(no_label_dir, "images")
    with_label_img_dir = os.path.join(with_label_dir, "images")
    with_label_lbl_dir = os.path.join(with_label_dir, "labels")
    
    # 建立目標資料夾
    os.makedirs(no_label_img_dir, exist_ok=True)
    os.makedirs(with_label_img_dir, exist_ok=True)
    os.makedirs(with_label_lbl_dir, exist_ok=True)

    # 收集所有的 images 和 labels 的 basename
    img_files = []
    if os.path.exists(img_dir):
        img_files = [f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')]
        
    lbl_files = []
    if os.path.exists(lbl_dir):
        lbl_files = [f for f in os.listdir(lbl_dir) if f.lower().endswith('.txt')]
        
    img_basenames = set([os.path.splitext(f)[0] for f in img_files])
    lbl_basenames = set([os.path.splitext(f)[0] for f in lbl_files])
    
    # 聯集所有檔名
    all_basenames = img_basenames.union(lbl_basenames)

    moved_no_label = 0
    moved_with_label = 0
    deleted_txt = 0

    print("開始清洗資料...")
    
    for basename in all_basenames:
        has_jpg = basename in img_basenames
        has_txt = basename in lbl_basenames
        
        jpg_path = os.path.join(img_dir, f"{basename}.jpg")
        txt_path = os.path.join(lbl_dir, f"{basename}.txt")
        
        # 動作 1: 把沒有對應標註檔的圖片獨立出來
        if has_jpg and not has_txt:
            dst = os.path.join(no_label_img_dir, f"{basename}.jpg")
            shutil.move(jpg_path, dst)
            moved_no_label += 1
            
        # 動作 2: 有標註檔的圖片也獨立出來 (連同標註檔一起移動)
        elif has_jpg and has_txt:
            dst_jpg = os.path.join(with_label_img_dir, f"{basename}.jpg")
            dst_txt = os.path.join(with_label_lbl_dir, f"{basename}.txt")
            shutil.move(jpg_path, dst_jpg)
            shutil.move(txt_path, dst_txt)
            moved_with_label += 1
            
        # 動作 3: 如果有標註檔但沒有圖片就刪除標註檔
        elif not has_jpg and has_txt:
            os.remove(txt_path)
            deleted_txt += 1

    print("資料清洗完成！")
    print(f"1. 移至 no_label/images 的無標註圖片數量: {moved_no_label}")
    print(f"2. 移至 with_label 的有標註圖片數量: {moved_with_label}")
    print(f"3. 刪除的孤兒標註檔 (.txt) 數量: {deleted_txt}")

if __name__ == "__main__":
    main()
