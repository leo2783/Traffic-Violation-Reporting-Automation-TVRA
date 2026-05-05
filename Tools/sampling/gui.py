import sys
import os
import logging
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QFileDialog, QDoubleSpinBox, 
    QComboBox, QCheckBox, QProgressBar, QTextEdit, QGroupBox, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject

# 導入去重服務
from main import DeduplicationService

class GUILogHandler(logging.Handler, QObject):
    """將 logging 輸出轉向 PyQt Signal 的 Handler"""
    log_signal = pyqtSignal(str)

    def __init__(self):
        super(GUILogHandler, self).__init__()
        QObject.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        self.log_signal.emit(msg)

class DeduplicationWorker(QThread):
    """處理去重邏輯的背景執行緒"""
    finished = pyqtSignal(bool, str)
    
    def __init__(self, input_folder, output_folder, threshold, yolo_weights, use_confidence, write_mode):
        super().__init__()
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.threshold = threshold
        self.yolo_weights = yolo_weights
        self.use_confidence = use_confidence
        self.write_mode = write_mode

    def run(self):
        try:
            service = DeduplicationService(threshold=self.threshold, yolo_weights=self.yolo_weights)
            service.execute(
                input_folder=self.input_folder,
                output_folder=self.output_folder,
                use_confidence=self.use_confidence,
                write_mode=self.write_mode
            )
            self.finished.emit(True, "去重任務已成功完成！")
        except Exception as e:
            self.finished.emit(False, f"執行發生錯誤: {str(e)}")

class SamplingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TVRA Sampling Tool - 圖片去重工具")
        self.setMinimumSize(700, 600)
        
        self.init_ui()
        self.setup_logging()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- 路徑選擇區 ---
        path_group = QGroupBox("路徑設定")
        path_layout = QVBoxLayout()
        
        # 輸入路徑
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("輸入資料夾:"))
        self.input_edit = QLineEdit()
        input_layout.addWidget(self.input_edit)
        btn_browse_input = QPushButton("瀏覽")
        btn_browse_input.clicked.connect(self.browse_input)
        input_layout.addWidget(btn_browse_input)
        path_layout.addLayout(input_layout)

        # 輸出路徑
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("輸出資料夾:"))
        self.output_edit = QLineEdit()
        output_layout.addWidget(self.output_edit)
        btn_browse_output = QPushButton("瀏覽")
        btn_browse_output.clicked.connect(self.browse_output)
        output_layout.addWidget(btn_browse_output)
        path_layout.addLayout(output_layout)
        
        path_group.setLayout(path_layout)
        main_layout.addWidget(path_group)

        # --- 參數設定區 ---
        param_group = QGroupBox("去重參數")
        param_layout = QHBoxLayout()
        
        param_layout.addWidget(QLabel("相似度閾值:"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(0.90)
        param_layout.addWidget(self.threshold_spin)
        
        param_layout.addSpacing(20)
        param_layout.addWidget(QLabel("取樣方式:"))
        self.sample_way_combo = QComboBox()
        self.sample_way_combo.addItems(["negative", "positive"])
        param_layout.addWidget(self.sample_way_combo)
        
        param_layout.addSpacing(20)
        param_layout.addWidget(QLabel("寫入模式:"))
        self.write_mode_combo = QComboBox()
        self.write_mode_combo.addItems(["per-folder", "per-video", "per-frame"])
        param_layout.addWidget(self.write_mode_combo)
        
        param_group.setLayout(param_layout)
        main_layout.addWidget(param_group)

        # --- YOLO 增強設定 ---
        yolo_group = QGroupBox("YOLO 信心度優化 (選填)")
        yolo_layout = QVBoxLayout()
        
        self.use_yolo_check = QCheckBox("啟用 YOLO 信心度排序策略")
        self.use_yolo_check.stateChanged.connect(self.toggle_yolo_ui)
        yolo_layout.addWidget(self.use_yolo_check)
        
        yolo_path_layout = QHBoxLayout()
        yolo_path_layout.addWidget(QLabel("YOLO 權重路徑:"))
        self.yolo_edit = QLineEdit()
        self.yolo_edit.setEnabled(False)
        yolo_path_layout.addWidget(self.yolo_edit)
        self.btn_browse_yolo = QPushButton("瀏覽")
        self.btn_browse_yolo.setEnabled(False)
        self.btn_browse_yolo.clicked.connect(self.browse_yolo)
        yolo_path_layout.addWidget(self.btn_browse_yolo)
        yolo_layout.addLayout(yolo_path_layout)
        
        yolo_group.setLayout(yolo_layout)
        main_layout.addWidget(yolo_group)

        # --- 執行區 ---
        exec_layout = QVBoxLayout()
        self.run_button = QPushButton("開始執行去重")
        self.run_button.setMinimumHeight(40)
        self.run_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.run_button.clicked.connect(self.start_deduplication)
        exec_layout.addWidget(self.run_button)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0) # 顯示忙碌狀態
        self.progress_bar.hide()
        exec_layout.addWidget(self.progress_bar)
        main_layout.addLayout(exec_layout)

        # --- 日誌輸出區 ---
        log_group = QGroupBox("執行狀態與日誌")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; font-family: Consolas;")
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)

    def setup_logging(self):
        self.log_handler = GUILogHandler()
        self.log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.log_handler.log_signal.connect(self.append_log)
        
        # 取得 root logger 或特定 logger 並加入 handler
        root_logger = logging.getLogger()
        root_logger.addHandler(self.log_handler)
        root_logger.setLevel(logging.INFO)

    def append_log(self, message):
        self.log_text.append(message)
        # 自動捲動到底部
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def browse_input(self):
        directory = QFileDialog.getExistingDirectory(self, "選擇輸入資料夾")
        if directory:
            self.input_edit.setText(directory)

    def browse_output(self):
        directory = QFileDialog.getExistingDirectory(self, "選擇輸出資料夾")
        if directory:
            self.output_edit.setText(directory)

    def browse_yolo(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "選擇 YOLO 權重檔案", "", "Weights (*.pt *.engine);;All Files (*)")
        if file_path:
            self.yolo_edit.setText(file_path)

    def toggle_yolo_ui(self, state):
        enabled = (state == Qt.CheckState.Checked.value)
        self.yolo_edit.setEnabled(enabled)
        self.btn_browse_yolo.setEnabled(enabled)

    def start_deduplication(self):
        input_path = self.input_edit.text()
        output_path = self.output_edit.text()
        
        if not input_path or not output_path:
            QMessageBox.warning(self, "警告", "請先設定輸入與輸出資料夾路徑！")
            return
            
        if self.use_yolo_check.isChecked() and not self.yolo_edit.text():
            QMessageBox.warning(self, "警告", "已啟用 YOLO 策略但未設定權重路徑！")
            return

        # 鎖定介面
        self.run_button.setEnabled(False)
        self.run_button.setText("執行中...")
        self.progress_bar.show()
        
        # 建立並啟動執行緒
        self.worker = DeduplicationWorker(
            input_folder=input_path,
            output_folder=output_path,
            threshold=self.threshold_spin.value(),
            yolo_weights=self.yolo_edit.text() if self.use_yolo_check.isChecked() else None,
            use_confidence=self.use_yolo_check.isChecked(),
            write_mode=self.write_mode_combo.currentText()
        )
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_finished(self, success, message):
        self.run_button.setEnabled(True)
        self.run_button.setText("開始執行去重")
        self.progress_bar.hide()
        
        if success:
            QMessageBox.information(self, "成功", message)
        else:
            QMessageBox.critical(self, "錯誤", message)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SamplingGUI()
    window.show()
    sys.exit(app.exec())
