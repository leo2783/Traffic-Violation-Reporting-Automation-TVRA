# Sampling 模組技術實現細節

## 概述
Sampling 模組 (`sampling/`) 的核心目的是透過深度學習特徵與物體檢測技術，自動化過濾與清洗行車紀錄器中擷取的大量重複或相似圖片。該模組主要由 `main.py` 與 `embedding.py` (包含 `ImageDeduplicator` 類別) 組成。

## 演算法流程

此模組實作了兩階段的圖片篩選機制：

```mermaid
graph TD
    A[開始: 輸入圖片路徑清單] --> B{是否啟用信心度採樣?}
    
    B -- 是 (Confident Sample) --> C[YOLO 推論]
    C --> C1[取得預測信心度與 Box 數量]
    C1 --> C2[根據正向/負向排序]
    C2 --> D
    
    B -- 否 --> D[MobileNetV3 特徵提取]
    
    D --> E[轉換為 NumPy Array 進行特徵扁平化]
    E --> F[計算餘弦相似度矩陣]
    
    F --> G{是否有提供 Box Counts?}
    G -- 是 --> H[比對兩張圖片的 Box 數量]
    H --> I{Box 數量相同?}
    I -- 否 --> J[強制將相似度歸零]
    I -- 是 --> K
    G -- 否 --> K[保留原始相似度]
    J --> K
    
    K --> L{相似度 > 閾值 (預設 0.95)?}
    L -- 是 --> M[標記為重複並過濾]
    L -- 否 --> N[保留為獨立圖片]
    
    M --> O[輸出最終保留清單]
    N --> O
    O --> P[結束]
```

## 核心技術細節

### 1. YOLO 信心度採樣 (Confident Sample)
- **模型使用**：載入訓練好的 YOLOv4 模型 (針對車牌/違規行為)。
- **邏輯**：預先對清單中的每一張圖片進行預測，獲取其檢測框的信心度。若需要保留高信心度的樣本（`positive`），則根據信心度進行升序排列（確保在後續的去重過程中，較高信心度的圖片會被保留）。

### 2. 特徵向量提取 (Feature Extraction)
- **模型選擇**：採用輕量級的 `MobileNet_V3_Small`，不僅速度快，且能有效捕捉圖片的高維度語義特徵。
- **處理方式**：
  - 將分類層 (Classifier) 替換為 `nn.Identity()`，使其直接輸出 1D 的 Embedding 向量。
  - 對輸入圖片進行 Pad (填充至 224x224) 與正規化 (Normalization)。

### 3. 餘弦相似度與強制過濾 (Cosine Similarity & Forced Filtering)
- **矩陣運算**：使用 NumPy 的矩陣乘法快速計算所有圖片特徵之間的餘弦相似度 (Cosine Similarity)。
- **標註數量保護機制**：若兩張圖片的相似度極高，但 YOLO 預測出的標註數量 (Box Counts) 不同，演算法會透過 Boolean Mask 強制將它們的相似度歸零。這能避免「背景相似，但其中一張有車、另一張沒車」的誤刪情況。

## 執行與使用
模組入口點為 `main.py`。執行時，程式會讀取指定資料夾，透過 `ImageDeduplicator().process_batch()` 取得過濾後的路徑清單，並將保留下來的圖片複製到新的 `cleaned_images` 資料夾中。