import os
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

def save_buffered_data(data, categories_dir):
    """處理並儲存單幀資料，完全無迴圈"""
    if not data:
        return 0
        
    cat = data['category']
    base_fname = data['base_fname']
    save_path = categories_dir[cat]
    
    # 存圖
    cv2.imwrite(os.path.join(save_path, f"{base_fname}.jpg"), data['img'])
    
    # 存標註檔 (使用 NumPy 矩陣化操作取代原本的 for 迴圈)
    if cat != "negative_sample" and len(data['cls']) > 0:
        txt_path = os.path.join(save_path, f"{base_fname}.txt")
        # 將 cls (N,) 與 xywhn (N, 4) 橫向合併為 (N, 5) 矩陣，並一次性寫入
        yolo_data = np.column_stack((data['cls'], data['xywhn']))
        np.savetxt(txt_path, yolo_data, fmt='%d %.6f %.6f %.6f %.6f')
        
    return 1

def main():
    model_path = r"C:/Users/qet63/Documents/Traffic-Violation-Reporting-Automation-TVRA-/TaiwanLicensePlate/YOLO/Detection/YOLO_V5_Result/V5_test1_detection_1280/weights/V5_test1_detection_1280.engine"
    video_dir = r"C:/Users/qet63/Videos/Movie"
    
    if not (os.path.exists(model_path) and os.path.exists(video_dir)):
        print("模型或影片路徑不存在")
        return

    # 建立輸出資料夾
    categories_dir = {
        cat: os.path.join("to_check", cat) 
        for cat in ["negative_sample", "to_review", "auto_labeled"]
    }
    for path in categories_dir.values():
        os.makedirs(path, exist_ok=True)

    video_list = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.ts'))]
    model = YOLO(model_path)

    for video_name in video_list:
        video_path = os.path.join(video_dir, video_name)
        video_basename = os.path.splitext(video_name)[0]
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        saved_count = 0
        current_sec_idx = 0
        best_frame = None 

        print(f"正在處理: {video_name}")
        results = model.predict(source=video_path, stream=True, vid_stride=10, conf=0.4, verbose=False)
        
        expected_steps = max(1, total_frames // 10)
        
        with tqdm(total=expected_steps, desc=f"處理 {video_name}", unit="step") as pbar:
            for idx, r in enumerate(results):
                pbar.update(1)
                if len(r.boxes) == 0:
                    continue
                
                frame_count = idx * 10
                sec_idx = int(frame_count // fps)
                
                # 跨越秒數區間，觸發儲存並清空暫存
                if sec_idx > current_sec_idx:
                    saved_count += save_buffered_data(best_frame, categories_dir)
                    best_frame = None
                    current_sec_idx = sec_idx
                    
                max_conf = float(r.boxes.conf.max())
                
                # 信心度分級
                if max_conf < 0.55: cat = "negative_sample"
                elif max_conf < 0.75: cat = "to_review"
                else: cat = "auto_labeled"
                
                # 競爭該秒的最高信心度代表
                if not best_frame or max_conf > best_frame['max_conf']:
                    best_frame = {
                        'max_conf': max_conf,
                        'category': cat,
                        'base_fname': f"{video_basename}_frame_{frame_count}",
                        'img': r.orig_img.copy(),
                        'cls': r.boxes.cls.cpu().numpy(),
                        'xywhn': r.boxes.xywhn.cpu().numpy()
                    }
                
                
        # 處理影片最後一秒的殘留資料
        saved_count += save_buffered_data(best_frame, categories_dir)
        print(f"影片 {video_name} 完成，共提取 {saved_count} 張圖片。\n")

if __name__ == "__main__":
    main()