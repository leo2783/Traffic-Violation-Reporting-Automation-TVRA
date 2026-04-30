# Pipeline Technical Details (PIPELINE_DETAILS_en.md)

This document provides in-depth technical information about the automated sampling pipeline implemented in `pipeline.py`.

---

## Architecture Overview

The pipeline follows a strictly **Object-Oriented (OO)** design to ensure modularity, maintainability, and scalability.

- **`BaseDataset`**: An abstract interface for different data sources.
- **`YoutubeDataset`**: Handles automated video link scraping from YouTube using Selenium.
- **`LocalDataset`**: Manages scanning of local video and image files.
- **`FeatureExtractor`**: Utilizes a pre-trained **MobileNetV3 Small** model to convert images into high-dimensional feature vectors.
- **`SamplingPipeline`**: The core controller orchestrating the entire process.
- **`AutoLabelClassifier`**: An independent module for clustering and selecting high-confidence samples.

---

## Data Acquisition

### YouTube Streaming
Instead of downloading entire videos, the pipeline uses `yt-dlp` to extract direct stream URLs. This avoids the common "Waiting for stream 0" issue and significantly reduces disk usage.

---

## Inference and Extraction

### 5 FPS Strategy
The pipeline uses OpenCV to process video streams at a rate of **5 frames per second (5 FPS)**. This ensures sufficient coverage of traffic events without generating excessive redundant data.

### Confidence Filtering
Detections are categorized based on their maximum confidence score:
- **Auto Labeled (`max_conf >= 0.8`)**: High-confidence frames are candidates for automated labeling. Bounding boxes are also filtered at the 0.8 threshold.
- **Negative Sample (`0.2 < max_conf < 0.65`)**: Hard negative candidates that deceive the model into low-to-mid confidence detections.

---

## Intelligent Sampling Logic

### Temporal Spacing (The 3-Second Rule)
To prevent the candidate pool from being dominated by near-identical consecutive frames, a **3-second (15 processed frames) interval** is strictly enforced for both auto-labeled and negative candidates within the same video.

### Performance & Memory Optimization
The pipeline utilizes **disk-backed temporary storage** (`temp_candidates` folders). Candidate frames are saved to disk immediately, keeping only metadata in RAM. This allows the system to process hundreds of videos (e.g., the 200-video YouTube target) without running out of memory (OOM).

### K-Means Clustering & Z-Score Weighting
1. **Feature Extraction**: Candidate images are passed through MobileNetV3 to get high-dimensional descriptors.
2. **Clustering**: K-Means groups images into scenes (e.g., rainy night, high-speed bridge, urban intersection).
3. **Difficulty Metric**: For negative samples, the mean confidence of each cluster represents its "deceptive power."
4. **Weighted Sampling**: Z-scores are calculated for these metrics and converted into probability weights $w$ (using a softmax-like distribution). This prioritizes sampling from more "difficult" scene types while maintaining global diversity.

---

## Modularization

### Standalone `auto.py`
The auto-label refinement logic is decoupled. The `auto.py` utility can be used independently to scan any folder of images, re-run YOLO inference to gather confidence metadata, and then apply the same K-Means top-10 selection logic to produce a high-quality, diverse dataset.
