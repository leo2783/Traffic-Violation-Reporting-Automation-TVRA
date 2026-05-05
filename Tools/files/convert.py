from ultralytics import YOLO

def ONXX(model):
    model.export(format='onnx',dynamic=True,half=True)

def TensorRT(model):
    model.export(format='engine',imgsz=960,dynamic=True,half=True,device='0',batch=4,simplify=True)

model=YOLO(r"C:/Users/qet63/Documents/Traffic-Violation-Reporting-Automation-TVRA-/TaiwanLicensePlate/YOLO\Detection/YOLO_V5_Result/V5_test1_detection_1280/weights/V5_test1_detection_1280.pt")
TensorRT(model)