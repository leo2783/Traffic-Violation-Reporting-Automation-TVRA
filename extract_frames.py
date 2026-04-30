import os
import cv2
from ultralytics import YOLO

def main():
    # 載入模型
    model_path = r"taiwan_plate/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"找不到模型檔案: {model_path}")
        return
        
    model = YOLO(model_path)
    
    video_dir = r"test_video"
    if not os.path.exists(video_dir):
        print(f"找不到影片資料夾: {video_dir}")
        return
        
    # 建立輸出資料夾結構
    base_out_dir = r"to_check"
    categories = {
        "negative_sample": os.path.join(base_out_dir, "negative_sample"),
        "to_review": os.path.join(base_out_dir, "to_review"),
        "auto_labeled": os.path.join(base_out_dir, "auto_labeled")
    }
    
    for path in categories.values():
        os.makedirs(path, exist_ok=True)
        
    video_list = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.ts'))]
    print("Start processing videos!!")
    
    for video_name in video_list:
        video_path = os.path.join(video_dir, video_name)
        print(f"正在處理: {video_name}")
        saved_count = 0
        
        # 使用 YOLO 內建的影片推論，設定 stream=True 避免記憶體溢出，vid_stride=10 跳過不必要的解碼
        results = model.predict(source=video_path, stream=True, vid_stride=10, conf=0.3, verbose=False)
        
        for idx, r in enumerate(results):
            frame_count = idx * 10
            
            # 若無預測框，則提早結束該回合
            if len(r.boxes) == 0:
                continue
                
            confs = r.boxes.conf.cpu().numpy()
            if len(confs) == 0:
                continue
                
            max_conf = float(max(confs))
            
            # 根據最大信心度分類
            category = None
            if 0.3 <= max_conf < 0.5:
                category = "negative_sample"
            elif 0.5 <= max_conf < 0.7:
                category = "to_review"
            elif max_conf >= 0.7:
                category = "auto_labeled"
                
            if category:
                video_basename = os.path.splitext(video_name)[0]
                base_filename = f"{video_basename}_frame_{frame_count}"
                
                # 儲存圖片
                img_save_path = os.path.join(categories[category], f"{base_filename}.jpg")
                cv2.imwrite(img_save_path, r.orig_img)
                
                # 若為 negative_sample，不留標註檔
                if category != "negative_sample":
                    boxes_to_save = []
                    xywhn = r.boxes.xywhn.cpu().numpy()
                    cls_indices = r.boxes.cls.cpu().numpy()
                    
                    for i in range(len(confs)):
                        c = int(cls_indices[i])
                        x, y, w, h = xywhn[i]
                        boxes_to_save.append(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                        
                    txt_save_path = os.path.join(categories[category], f"{base_filename}.txt")
                    with open(txt_save_path, 'w', encoding='utf-8') as f:
                        if boxes_to_save:
                            f.write('\n'.join(boxes_to_save))
                            
                saved_count += 1
                
        print(f"影片 {video_name} 處理完成，共提取並儲存了 {saved_count} 張照片。")

    print("\n所有影片處理完成！檔案皆已分類存入 to_check 資料夾。")

if __name__ == "__main__":
    main()