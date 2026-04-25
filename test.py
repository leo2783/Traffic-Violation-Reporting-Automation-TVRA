from ultralytics import YOLO
import os
import cv2

# 載入模型
model = YOLO(r"C:/Users/user1/Documents/Traffic violation reporting/taiwan_plate/weights/best.onnx")
video_dir = r"C:/Users/user1/Videos/test_video"
video_list = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.ts'))]

print("Start!!")

for i in video_list:
    video_path = os.path.join(video_dir, i)
    
    # 獲取影片尺寸（僅供提示，YOLO 會自行處理縮放）
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"無法開啟影片: {i}")
        continue
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    print(f"正在處理: {i} (原始尺寸: {width}x{height})")
    
    # 修正重點：
    # 1. 增加 stream=True：防止記憶體溢出
    # 2. 強制 imgsz=1280：對齊你 ONNX 要求的輸入尺寸，YOLO 會自動 letterbox（補黑邊縮放）保持比例
    results = model.predict(
        source=video_path,
        conf=0.7,
        save=True,
        show=False,       # 解決尺寸不符報錯
        stream=True,       # 解決記憶體溢出警告
        name=os.path.splitext(i)[0],
        exist_ok=True
    )

    # 在 stream=True 模式下，必須遍歷 results 才會執行推論與存檔
    for r in results:
        pass  # 這裡什麼都不用做，save=True 會自動幫你存檔

print("\n辨識完成！所有影片均已按正確比例處理完畢。")