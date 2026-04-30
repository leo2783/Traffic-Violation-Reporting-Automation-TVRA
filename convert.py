from ultralytics import YOLO

def ONXX(model):
    model.export(format='onnx',dynamic=True,half=True)

def TensorRT(model):
    model.export(format='engine',imgsz=1280,dynamic=True,half=True,device='0',batch=2,simplify=True)

model=YOLO(r"C:/Users/qet63/Documents/Traffic-Violation-Reporting-Automation-TVRA-/YOLO_V4_Result/train/weights/best.pt")
TensorRT(model)