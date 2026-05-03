#實作要求
對於一群不知道信心度的已去重負樣本集進行抽樣，流程為參考embedding.py的_extract_embedding進行維度轉換，特偵向量的轉換為同一模型，再用UMAP 降維與 HDBSCAN 分群 (Clustering)，用best.pt進行predict，得到每張圖片的信心度，計算每群的平均信心度，用數學方法轉換為被抽取機率，最後實作隨機抽取(信心度越高代表機率越高)，並把結果存到negative_result。
程式名稱為：extract_negative.py
#實作要求
遵守物件導向規則，讓使用者在main.py實現的時候有好的API接口，也確保資料和類別不會被惡意修改。
#API
允許使用者輸入指定的輸出folder，若沒有輸入則為預設，強制使用者輸入需要抽取的樣本數。強制使用者輸入要抽取的path。
#安全守則
你必須要確保指令的合法性。 