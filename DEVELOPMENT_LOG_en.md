# Traffic Violation Reporting Automation (TVRA) - Development Log

This document details the development journey of this project, including model version evolution, data processing specifics, and technical challenges encountered along with their solutions.

---

## 2026-05-04 to 2026-05-05
**Multi-resolution Model Training and Labeling Issue Resolution**
- **Multi-resolution Model Training**: Conducted training tests with different resolutions using different hardware:
  - Trained `test1` at `1280` resolution using an RTX 4090.
  - Trained `test2` at `960` resolution using an RTX 5070.
- **Poly Image Annotation Issue**: Discovered labeling errors during the training phase. Pre-training checks via `check_label.py` confirmed that a total of 79 Poly images in the dataset had invalid annotations (64 in train, 15 in val; specifically self-intersecting polygon segments). **An Issue has been posted to track and resolve this problem**.

## 2026-05-01 to 2026-05-03
**Automated Deduplication Filtering Mechanism (Sampling)**
- **High-Dimensional Feature Extraction & Deduplication**: Developed the `sampling` module, integrating MobileNetV3 to convert images into high-dimensional feature vectors (Embeddings). To achieve high-speed computation, PyTorch is utilized for GPU-accelerated tensor matrix operations, calculating a Cosine Similarity matrix to accurately filter out redundant frames. For detailed technical implementations and algorithm workflows, please refer to: [Sampling Module Implementation Details](./Tools/sampling/detail/SAMPLING_DETAILS_en.md).
- **Multi-layered Filtering Strategy**: Discarded the planned K-Means clustering in favor of **Bounding Box Count Comparison** and **YOLO Confidence Sorting**. The system only flags images as duplicates when both box counts are identical and feature similarity meets the threshold, ensuring rigorous and high-quality data cleaning.
- **Data Cleaning & Dataset Organization**: The design of the data cleaning and deduplication algorithm continued through May 3. The comprehensive organization of the dataset was fully completed on May 3.
- **Future Work**:
  - Explore the implementation of "Negative Sample Mining based on Contrastive Learning" to improve the performance of advanced retrieval models (Reference: Research by Tsai-Tsung, Chen).

## 2026-04-28 to 2026-04-30
**Model Optimization, Hardware Acceleration, and Deployment Testing**
- **In Progress**: Data cleaning preparations using `yolo26s-seg`.

### 2026-04-28: TensorRT Hardware Acceleration Implementation
- **Technology Selection**: To overcome performance bottlenecks on the local RTX 3050 (4GB) when running high-resolution models, NVIDIA TensorRT was introduced for hardware acceleration.
- **ONNX Conversion & Optimization**:
  - Implemented `convert.py` script to convert Version 4 weights (`best.pt`) to ONNX format using official tools.
  - Enabled `dynamic=True` for dynamic aspect ratio support and `half=True` for FP16 half-precision computation.
- **TensorRT Engine Compilation**:
  - Compiled a specialized `.engine` serialized file for the RTX 3050 GPU architecture.
  - **Core Parameters**: Set `imgsz=1280` to preserve pixel details of distant license plates; configured `simplify=True` for graph optimization; set `batch=2` to balance inference latency and throughput.
- **Performance Verification**: Inference latency dropped significantly after switching to the Engine file, successfully overcoming local VRAM limitations.

### 2026-04-29 to 2026-04-30: Integration Testing and Workflow Consolidation
- **Pipeline Integration**: Formally integrated the TensorRT-accelerated model into the automated identification workflow.
- **Stability Testing**: Stress-tested the system with various dashcam footage lengths to ensure no VRAM overflow or performance degradation over long runs.
- **Logic Adjustments**: Adjusted post-processing NMS thresholds and coordinate transformation logic based on the model's exported Tensor structure.

## 2026-04-27
**Version 4 Training and Gradient Explosion Troubleshooting**
- **Dataset Verification & Cleaning**: Performed deep verification of image-label correspondence and implemented stricter dataset cleaning.
- **Model Training (Version 4)**: Encountered "Gradient Explosion" issues early in training, causing Loss divergence. Successfully resolved by adjusting hyperparameters and lowering the Learning Rate.
- **Results & Next Steps**: Successfully trained the Version 4 model. Plan to introduce a Segmentation model in the next phase to further enhance recognition precision in complex real-world road conditions.

## 2026-04-25 to 2026-04-26
**Version 3 Model Training and Performance Breakthrough**
- **Continuous Data Optimization**: Introduced cleaner datasets for training after several days of tool-assisted cleaning.
- **Cloud Training**: Continued using Google Colab for Version 3 model training.
- **Metric Breakthrough**: Significant performance gains on the validation set, with mAP@0.5 breaking the 0.92 mark. License plate and violation feature extraction became much more stable.

## 2026-04-22 to 2026-04-24
**Building Automated Toolchains and Large-scale Data Cleaning**
- **Data Quality Enhancement**: Launched a massive data cleaning process to improve model accuracy.
- **Tool Development (YoloTool)**: Developed custom scripts like `YoloTool` (e.g., `FileCompareTool`) to automate file management, batch clean invalid data, and sync images with labels. These scripts drastically reduced manual processing time.

## 2026-04-22
**Breaking Hardware Bottlenecks, Migrating to Cloud Computing**
- **Hardware Limitations**: Local RTX 3050 (4GB VRAM) proved insufficient for large-scale, deep model training.
- **Migration**: Decided to use Google Colab with A100 GPUs as the primary training environment.
- **First Cloud Training**: Completed a full training cycle on Colab in approximately 5 hours, yielding updated weights.

## 2026-04-21
**Version 2 Model Fine-tuning and Semi-automated Labeling Workflow**
- **Model Fine-tuning**: Used `yolo26n.pt` as the base model for fine-tuning, alongside `yolo26s.pt` for dual-track testing.
- **Data Augmentation & Semi-auto Labeling**:
  - Processed dashcam footage with frame extraction at 5 fps.
  - Used Version 1 model for initial inference and pre-labeling, followed by manual random sampling and correction to boost efficiency and accuracy.
- **File Management**: Completed extensive merging and restructuring of image and label directories.

## 2026-04-20
**Local Hardware Testing and Open Dataset Integration**
- **Hardware Validation**: Successfully ran initial ALPR model inference on local NVIDIA RTX 3050 (4GB).
- **Dataset Integration**: Integrated the [EZCon/taiwan-license-plate-recognition](https://huggingface.co/datasets/EZCon/taiwan-license-plate-recognition) open dataset from Hugging Face as the foundation for early training and validation.

## 2026-04-19
**Project Initialization and Architecture Establishment**
- **Project Launch**: Formally initialized the project, completing basic architecture planning and goal setting.
- **Technology Selection**: Confirmed YOLO for object detection (plates/violations) combined with OCR for text extraction.
- **Future Plans**: Focused next steps on dataset expansion and training a proprietary YOLO model.
