# YoloTool (Data Processing & Comparison Tool)

YoloTool is a Qt-based graphical application designed for processing, cleaning, and optimizing YOLO-format datasets.
This directory contains the compiled Windows executable `FileCompareTool.exe`, along with its required dynamic link libraries (DLLs) and dependencies.

## How to Run
Simply double-click `FileCompareTool.exe` to launch the graphical interface.

---

## Expected Folder Structure (Important)
To ensure the program can correctly read images and annotation files and avoid bugs, please make sure your selected source folder follows one of these two structures:

### 1. Standard YOLO Structure (Recommended)
The root directory contains `images` and `labels` folders. The program will automatically scan subdirectories (e.g., `train`, `val`) or read the files directly inside. Images and their corresponding `.txt` annotation files must share the exact same base name.

```text
Source Directory/
├── images/
│   ├── train/      (Optional subdirectory)
│   │   ├── image_001.jpg
│   │   └── image_002.png
│   └── val/        (Optional subdirectory)
│       └── image_003.jpg
└── labels/
    ├── train/      (Optional subdirectory)
    │   ├── image_001.txt
    │   └── image_002.txt
    └── val/        (Optional subdirectory)
        └── image_003.txt
```

### 2. Single-Level Flat Structure
If no `images` and `labels` directories are found, the program falls back to a single-level mode, treating your selected directory as the root for both images and annotations. This means your images and `.txt` annotation files must be placed together in the same folder.

```text
Source Directory/
├── image_001.jpg
├── image_001.txt
├── image_002.png
└── image_002.txt
```
> **Note**: Regardless of the structure, the annotation files must be in YOLO `.txt` format, and every valid annotation line must contain at least 5 values (Class ID and 4 coordinates).

---

## Architecture

![YoloTool Architecture](architecture.png)

## Key Features

The tool provides four main tabs at the top of the interface, automating different stages of YOLO data processing:

### 📁 1. File Comparison
Used to compare contents between multiple "Source Directories" and a single "Target Directory," offering advanced cleaning features.
*   **Comparison Mode**: Supports comparison by "Full Name", "Base Name", or "File Content (MD5)".
*   **Sync & Clean Invalid Annotations**: Automatically scans the target directory to delete invalid `.txt` annotation files (empty or without coordinates), along with their corresponding images.
*   **File Sync Operations**:
    *   Delete files in the target directory that match the source.
    *   Delete files in the source that "do not exist in the target," helping with data synchronization and filtering.

### 📊 2. Dataset Splitter
Splits structured images and annotations into standard YOLO training directory formats (`images/train`, `images/val`, `labels/train`, `labels/val`).
*   **Smart Detection**: Automatically detects whether the source is a standard YOLO structure or a single-level folder.
*   **Random Split**: Supports custom Train/Val ratios (default is 80% for training) and automatically shuffles the data randomly.
*   **Auto-Generate Config File**: Scans all parsed annotation files for Class IDs and automatically generates `dataset.yaml` in the output directory for immediate training use.

### 🖼️ 3. Negative Sample Extraction
Quickly filters out images "without annotation files" or with "empty annotations" from the dataset, extracting them as background negative samples.
*   **Auto-Generate Empty Annotations**: Can optionally generate corresponding empty `.txt` files to meet YOLO formatting requirements for negative sample training.
*   **Processing Modes**: Supports "Copy" or "Move" directly to the specified negative sample folder.

### 🎯 4. YOLO Cleaner & Renamer
Performs final cleanup, label modification, and file naming standardization on the YOLO dataset.
*   **Clean Empty Files**: Automatically deletes empty annotation files and their corresponding images.
*   **Batch Modify Class IDs**: Can unify all Class IDs in the annotation files within the directory to a specific number (e.g., change all to `0`).
*   **Smart Batch Renaming**:
    *   Supports custom prefixes (e.g., `image_`) and zero-padded numbering (e.g., `00001`).
    *   **Smart Continuation**: The program automatically finds the highest numbered file that matches the naming rule in the current folder, and renames "only files that don't match the rule" sequentially, preserving the filenames of already standardized data.

---

## Notes
*   Do not delete or move any `.dll` files or subdirectories (such as `platforms/`, `styles/`, etc.) within the `YoloTool` folder. These are essential dependencies for the Qt application, and removing them will prevent the program from launching.
*   The C++ source code and UI design files for this tool are by leo2783. https://github.com/leo2783
*   **Open Source License**: MIT License
*   **Developer Contact**: qet6322076690@gmail.com
