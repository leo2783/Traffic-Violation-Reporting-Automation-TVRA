# 目標
實現抽取樣本自動化流水線，程式要能提供使用youtube影片進行抽取還是本地運行，是照片還是影片。
# 可參考
## 檔案
extract_frame.py data_collect.py YOLO_V4_Result\train\weights
## 程式碼
```py
from ultralytics import YOLO

# Load a pretrained YOLO26n model
model = YOLO("yolo26n.pt")

# Define source as YouTube video URL
source = "https://youtu.be/LNwODJXcvt4"

# Run inference on the source
results = model(source, stream=True)  # generator of Results objects
```
# 實作要求
嚴格遵守物件導向
# 實作大綱
## 資料獲取
1. 參考data_collect.py實作行車紀錄器網頁抓取類別，用串流方式。
2. 提供使用者可以指定本地路徑的類別
3. 此類別封裝成dataset
## 推論擷取
1. 利用取的的資料進行取frame環節，若是影片，則利用OpenCV每秒抽5frame，在利用best.engine進行偵測，保留信心度大於0.2和小於0.65的照片。
2. 把信心度大於0.8的資料連同標註一同存到auto_labeled，檔案要分成labels跟images
## 特徵轉換
1. 將這些圖片透過輕量神經網路，轉換成高維度的特徵向量。
2. 模型選用torchvision.models.mobilenet_v3_small
## K-Means 分群
利用分類器，把相似的廠景結合。
## 定義難度指標
利用數學計算每一群的信心度，這會代表整體類群的欺騙性，指標越高代表該群有更強的欺騙性。
## Z-score機率抽樣
對於各類群的信心度指標，計算Z-score並轉化為機率權重 w。
## 資料抽取
根據w，隨機抽取資料作為negative sample

# 注意事項
1. 實作過程要確保資訊安全，要二次確認使用的終端機指令是否危害使用者
2. negative_sample不需要留label檔案。
3. 要創建一個叫做negative_clean的資料夾存放這些實作後的程式碼，裡面有negative_sample和auto_labeled