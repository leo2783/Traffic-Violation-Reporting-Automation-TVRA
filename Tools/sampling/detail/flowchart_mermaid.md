graph TD
    A[開始 Input Images] --> B[MobileNetV3 特徵擷取 Feature Extraction]
    B --> C[計算相似度 Calculate Similarity Tensor]
    C --> D{相似度閾值 Deduplication > 0.90}
    D -- Discard --> H[結束 Output Images]
    D -- Keep --> E[YOLO 推論 YOLO Inference Confidence]
    E --> F[UMAP + HDBSCAN 降維與分群]
    F --> G[依照 Softmax 機率進行抽樣 Sample by Prob]
    G --> H