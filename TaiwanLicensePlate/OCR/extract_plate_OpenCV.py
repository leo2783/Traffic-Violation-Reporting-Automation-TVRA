import cv2
from ultralytics import YOLO
from pathlib import Path

video_path="input.mp4"
model_path=r"C:/Users/qet63/Documents/Traffic-Violation-Reporting-Automation-TVRA-/TaiwanLicensePlate/YOLO/Detection/YOLO_V5_Result/V5_test1_detection_1280/weights/V5_test1_detection_1280.pt"
output_dir=Path("plates")
output_dir.mkdir(exist_ok=True)

model=YOLO(model_path)
cap=cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise IOError(f"Cannot open video: {video_path}")

frame_count=0
save_count=0

while True:
    ret, frame=cap.read()
    if not ret:
        break

    if frame_count % 5 == 0:  
        results=model(frame, verbose=False, conf=0.7)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2=map(int, box.xyxy[0])
                plate_img=frame[y1:y2, x1:x2]
                save_path=output_dir / f"plate_{save_count}.jpg"
                cv2.imwrite(str(save_path), plate_img)
                save_count += 1
    frame_count += 1
cap.release()
print(f"Extracted {save_count} license plate images to {output_dir}") 
