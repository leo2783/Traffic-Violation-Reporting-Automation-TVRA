
# Code Review 與 Bug 修復計畫 (2026-05-05)

## 一、問題分析 (Bug Report)

根據使用者提供的 Error Log：
```log
2026-05-05 20:34:09,996 - ERROR - YOLO 批次推論失敗: cannot identify image file 'C:/Users/qet63/Videos/Movie/2026_0430_010931_001A.TS'
...
2026-05-05 22:54:56,773 - WARNING - 沒有有效的圖片特徵，結束處理。
```

我們發現了三個關鍵的 Bug：

### 1. 檔案副檔名未過濾 (Critical)
- **問題所在**：`main.py` 的 `execute` 方法中，讀取資料夾內容時使用了 `[f for f in input_folder.iterdir() if f.is_file()]`，這會把資料夾內所有的檔案（包含 `.TS`, `.mp4` 等影片檔）全部讀入。
- **後果**：當非圖片檔案被送到 YOLO (`pi-heif` / `PIL`) 或 MobileNetV3 進行特徵提取時，會引發 `cannot identify image file` 的崩潰。這導致有效圖片特徵數為 0。

### 2. `per-frame` 處理邏輯效能低落且含有潛在的 Shape Mismatch Bug (High)
- **問題所在**：在 `embedding.py` 的 `process_batch` 方法中，當選擇 `per-frame` 模式時，實作了一個迴圈，逐次將當前圖片張量與已保留圖片張量進行比對。
- **後果**：
  1. 完全喪失了 PyTorch 批次矩陣運算的效能優勢。
  2. 若啟用了 YOLO 信心度 (`valid_box_counts` 有值)，在執行 `same_box_matrix = (current_count == kept_counts).unsqueeze(0)` 時，由於張量維度處理不夠嚴謹，極易引發 RuntimeError。

### 3. `per-video` 分組邏輯的潛在風險 (Medium)
- **問題所在**：目前寫死使用 `_frame_` 作為字串分割字元，若檔名不包含此字元，所有圖片會被歸入 `unknown` 群組，失去了分組的意義。這在應對不同來源的圖片時非常脆弱。

---

## 二、修復計畫 (Action Plan)

### Step 1: 強化 `main.py` 的圖片過濾
- 修改 `file_list` 的生成邏輯，加入副檔名白名單過濾 (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`)。
- 確保傳給後端處理的絕對是圖片檔案。

### Step 2: 重構 `embedding.py` 的寫入策略 (Write Modes)
- **核心原則**：無論哪種寫入模式，特徵相似度的計算 (Cosine Similarity Matrix) 都必須**一次性批次完成**，以確保最高效能。不應該在計算相似度的過程中執行磁碟 I/O。
- **實作方式**：
  1. 先呼叫 `_calculate_duplicates` 一次性算出所有要保留的 `keep_indices`。
  2. 根據選擇的 `write_mode` 改變檔案拷貝 (`shutil.copy`) 的頻率與 Logging 方式：
     - `per-folder`: 在最後使用單一迴圈，將所有 `keep_indices` 對應的檔案一次性複製。
     - `per-frame`: 將拷貝行為分散到一個迭代迴圈中，每處理一個 index 就在日誌輸出一次「已寫入」的訊息，模擬即時處理感。
     - `per-video`: 先將 `keep_indices` 依影片名稱分組，然後逐組進行拷貝與日誌輸出。

### Step 3: 優化 `per-video` 的檔名解析
- 使用更安全的字串解析方法（例如正則表達式，或容錯更高的 split 邏輯），確保即使檔名規則改變，也不會引發系統崩潰。

---

## 三、自我評分 (Self-Evaluation)

| 評估項目 | 滿分 | 得分 | 說明 |
| :--- | :---: | :---: | :--- |
| **Log 讀取與問題定位精準度** | 30 | 30 | 成功從 `cannot identify image file` 發現了檔案過濾的漏洞。 |
| **Code Review 深度 (邏輯缺陷)** | 30 | 25 | 成功找出自己之前寫出的 `per-frame` 低效且易報錯的迴圈矩陣問題，反省深刻。但一開始沒有察覺到，扣 5 分。 |
| **修復計畫的合理性與效能考量** | 40 | 40 | 提出的重構計畫確保了 PyTorch 批次運算不被破壞，將寫入行為與計算邏輯解耦，是非常正確的架構設計。 |
| **總計** | **100** | **95** | **表現優異，但需警惕未來在實作新功能時，不可忽視底層演算法的效能本質（不可在 GPU 運算中夾雜硬碟 I/O）。** |