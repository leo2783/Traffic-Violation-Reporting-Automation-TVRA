"""PyQt GUI workbench for the TVRA sampling module."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    from .services import (
        DeduplicationService,
        NegativeSamplingService,
        ValidationCleanService,
        YoloTestService,
    )
except ImportError:
    from services import (
        DeduplicationService,
        NegativeSamplingService,
        ValidationCleanService,
        YoloTestService,
    )


logger = logging.getLogger(__name__)


class GUILogHandler(logging.Handler, QObject):
    """Forward Python logging records to a Qt signal."""

    log_signal = pyqtSignal(str)

    def __init__(self) -> None:
        logging.Handler.__init__(self)
        QObject.__init__(self)

    def emit(self, record: logging.LogRecord) -> None:
        self.log_signal.emit(self.format(record))


class TaskWorker(QThread):
    """Generic background worker for long-running GUI tasks."""

    finished = pyqtSignal(bool, str)
    progress = pyqtSignal(int, int)

    def __init__(self, task_name: str, task: Callable[[], object]) -> None:
        super().__init__()
        self._task_name = task_name
        self._task = task

    def emit_progress(self, current: int, total: int) -> None:
        """Expose a callback-compatible progress emitter."""

        self.progress.emit(current, total)

    def run(self) -> None:
        try:
            result = self._task()
            suffix = ""
            if isinstance(result, list):
                suffix = f" 共處理 {len(result)} 筆結果。"
            self.finished.emit(True, f"{self._task_name} 已成功完成！{suffix}")
        except Exception as exc:  # noqa: BLE001 - GUI must surface task exceptions.
            logger.exception("%s 執行失敗", self._task_name)
            self.finished.emit(False, f"{self._task_name} 執行發生錯誤: {exc}")


class SamplingGUI(QMainWindow):
    """Multi-tab GUI workbench for sampling module workflows."""

    def __init__(self) -> None:
        super().__init__()
        self.worker: TaskWorker | None = None
        self.run_buttons: list[QPushButton] = []

        self.setWindowTitle("TVRA Sampling Tool - 資料工程工作台")
        self.setMinimumSize(960, 760)

        self.init_ui()
        self.setup_logging()

    def init_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_dedup_tab(), "圖片去重")
        self.tabs.addTab(self.create_negative_sampling_tab(), "負樣本抽樣")
        self.tabs.addTab(self.create_validation_clean_tab(), "驗證集清洗")
        self.tabs.addTab(self.create_yolo_test_tab(), "YOLO 測試")
        self.tabs.addTab(self.create_auto_label_tab(), "Auto Label")
        main_layout.addWidget(self.tabs)

        exec_group = QGroupBox("任務狀態")
        exec_layout = QVBoxLayout(exec_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        exec_layout.addWidget(self.progress_bar)
        main_layout.addWidget(exec_group)

        log_group = QGroupBox("執行狀態與日誌")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(
            "background-color: #1e1e1e; color: #d4d4d4; font-family: Consolas;"
        )
        log_layout.addWidget(self.log_text)
        main_layout.addWidget(log_group)

    def create_dedup_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        path_group = QGroupBox("路徑設定")
        path_layout = QVBoxLayout(path_group)
        self.dedup_input_edit = self.add_path_row(path_layout, "輸入圖片資料夾:", "folder")
        self.dedup_output_edit = self.add_path_row(path_layout, "輸出資料夾:", "folder")
        layout.addWidget(path_group)

        param_group = QGroupBox("去重參數")
        param_layout = QHBoxLayout(param_group)
        param_layout.addWidget(QLabel("相似度閾值:"))
        self.dedup_threshold_spin = QDoubleSpinBox()
        self.dedup_threshold_spin.setRange(0.0, 1.0)
        self.dedup_threshold_spin.setSingleStep(0.05)
        self.dedup_threshold_spin.setValue(0.90)
        param_layout.addWidget(self.dedup_threshold_spin)

        param_layout.addWidget(QLabel("取樣方式:"))
        self.dedup_sample_way_combo = QComboBox()
        self.dedup_sample_way_combo.addItems(["negative", "positive"])
        param_layout.addWidget(self.dedup_sample_way_combo)

        param_layout.addWidget(QLabel("寫入模式:"))
        self.dedup_write_mode_combo = QComboBox()
        self.dedup_write_mode_combo.addItems(["per-folder", "per-video", "per-frame"])
        param_layout.addWidget(self.dedup_write_mode_combo)
        layout.addWidget(param_group)

        yolo_group = QGroupBox("YOLO 信心度優化")
        yolo_layout = QVBoxLayout(yolo_group)
        self.dedup_use_yolo_check = QCheckBox("啟用 YOLO 信心度排序策略")
        yolo_layout.addWidget(self.dedup_use_yolo_check)
        self.dedup_yolo_edit = self.add_path_row(
            yolo_layout,
            "YOLO 權重路徑:",
            "file",
            "Weights (*.pt *.engine);;All Files (*)",
        )
        self.dedup_yolo_edit.setEnabled(False)
        self.dedup_use_yolo_check.stateChanged.connect(
            lambda state: self.dedup_yolo_edit.setEnabled(state == Qt.CheckState.Checked.value)
        )
        layout.addWidget(yolo_group)

        run_button = QPushButton("開始執行圖片去重")
        run_button.clicked.connect(self.start_deduplication)
        self.register_run_button(run_button)
        layout.addWidget(run_button)
        layout.addStretch()
        return tab

    def create_negative_sampling_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        path_group = QGroupBox("路徑設定")
        path_layout = QVBoxLayout(path_group)
        self.negative_input_edit = self.add_path_row(path_layout, "輸入圖片資料夾:", "folder")
        self.negative_output_edit = self.add_path_row(path_layout, "輸出資料夾:", "folder")
        self.negative_yolo_edit = self.add_path_row(
            path_layout,
            "YOLO 權重路徑:",
            "file",
            "Weights (*.pt *.engine);;All Files (*)",
        )
        layout.addWidget(path_group)

        param_group = QGroupBox("抽樣參數")
        param_layout = QHBoxLayout(param_group)
        param_layout.addWidget(QLabel("抽樣數量:"))
        self.negative_num_spin = QSpinBox()
        self.negative_num_spin.setRange(1, 1_000_000)
        self.negative_num_spin.setValue(100)
        param_layout.addWidget(self.negative_num_spin)

        param_layout.addWidget(QLabel("Temperature:"))
        self.negative_temperature_spin = QDoubleSpinBox()
        self.negative_temperature_spin.setRange(0.1, 100.0)
        self.negative_temperature_spin.setSingleStep(0.5)
        self.negative_temperature_spin.setValue(5.0)
        param_layout.addWidget(self.negative_temperature_spin)
        layout.addWidget(param_group)

        run_button = QPushButton("開始執行負樣本抽樣")
        run_button.clicked.connect(self.start_negative_sampling)
        self.register_run_button(run_button)
        layout.addWidget(run_button)
        layout.addStretch()
        return tab

    def create_validation_clean_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        path_group = QGroupBox("路徑設定")
        path_layout = QVBoxLayout(path_group)
        self.val_source_edit = self.add_path_row(path_layout, "來源圖片資料夾:", "folder")
        self.val_output_edit = self.add_path_row(path_layout, "輸出 dataset 資料夾:", "folder")
        self.val_yolo_edit = self.add_path_row(
            path_layout,
            "YOLO 權重路徑:",
            "file",
            "Weights (*.pt *.engine);;All Files (*)",
        )
        layout.addWidget(path_group)

        param_group = QGroupBox("清洗參數")
        param_layout = QHBoxLayout(param_group)
        param_layout.addWidget(QLabel("信心度門檻:"))
        self.val_threshold_spin = QDoubleSpinBox()
        self.val_threshold_spin.setRange(0.0, 1.0)
        self.val_threshold_spin.setSingleStep(0.05)
        self.val_threshold_spin.setValue(0.60)
        param_layout.addWidget(self.val_threshold_spin)
        layout.addWidget(param_group)

        run_button = QPushButton("開始執行驗證集清洗")
        run_button.clicked.connect(self.start_validation_clean)
        self.register_run_button(run_button)
        layout.addWidget(run_button)
        layout.addStretch()
        return tab

    def create_yolo_test_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        path_group = QGroupBox("來源與輸出")
        path_layout = QVBoxLayout(path_group)

        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("來源類型:"))
        self.test_source_combo = QComboBox()
        self.test_source_combo.addItems(["image", "video", "file", "youtube"])
        self.test_source_combo.currentTextChanged.connect(self.update_yolo_test_source_ui)
        source_layout.addWidget(self.test_source_combo)
        path_layout.addLayout(source_layout)

        self.test_path_edit = self.add_path_row(path_layout, "本地路徑:", "dynamic")
        self.test_output_edit = self.add_path_row(path_layout, "輸出資料夾:", "folder")
        self.test_yolo_edit = self.add_path_row(
            path_layout,
            "YOLO 權重路徑:",
            "file",
            "Weights (*.pt *.engine);;All Files (*)",
        )
        layout.addWidget(path_group)

        param_group = QGroupBox("測試參數")
        param_layout = QHBoxLayout(param_group)
        param_layout.addWidget(QLabel("Conf:"))
        self.test_conf_spin = QDoubleSpinBox()
        self.test_conf_spin.setRange(0.0, 1.0)
        self.test_conf_spin.setSingleStep(0.05)
        self.test_conf_spin.setValue(0.70)
        param_layout.addWidget(self.test_conf_spin)

        param_layout.addWidget(QLabel("YouTube 數量:"))
        self.test_count_spin = QSpinBox()
        self.test_count_spin.setRange(1, 500)
        self.test_count_spin.setValue(5)
        param_layout.addWidget(self.test_count_spin)
        layout.addWidget(param_group)

        run_button = QPushButton("開始執行 YOLO 測試")
        run_button.clicked.connect(self.start_yolo_test)
        self.register_run_button(run_button)
        layout.addWidget(run_button)
        layout.addStretch()
        self.update_yolo_test_source_ui(self.test_source_combo.currentText())
        return tab

    def create_auto_label_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        path_group = QGroupBox("路徑設定")
        path_layout = QVBoxLayout(path_group)
        self.autolabel_input_edit = self.add_path_row(path_layout, "輸入圖片資料夾:", "folder")
        self.autolabel_output_edit = self.add_path_row(path_layout, "輸出資料夾:", "folder")
        self.autolabel_yolo_edit = self.add_path_row(
            path_layout,
            "YOLO 權重路徑:",
            "file",
            "Weights (*.pt *.engine);;All Files (*)",
        )
        layout.addWidget(path_group)

        param_group = QGroupBox("標註參數")
        param_layout = QHBoxLayout(param_group)
        
        param_layout.addWidget(QLabel("信心度閾值:"))
        self.autolabel_conf_spin = QDoubleSpinBox()
        self.autolabel_conf_spin.setRange(0.0, 1.0)
        self.autolabel_conf_spin.setSingleStep(0.05)
        self.autolabel_conf_spin.setValue(0.80)
        param_layout.addWidget(self.autolabel_conf_spin)

        param_layout.addWidget(QLabel("相似度閾值:"))
        self.autolabel_sim_spin = QDoubleSpinBox()
        self.autolabel_sim_spin.setRange(0.0, 1.0)
        self.autolabel_sim_spin.setSingleStep(0.05)
        self.autolabel_sim_spin.setValue(0.90)
        param_layout.addWidget(self.autolabel_sim_spin)
        
        layout.addWidget(param_group)

        output_group = QGroupBox("輸出選項")
        output_layout = QHBoxLayout(output_group)
        
        self.autolabel_copy_images_check = QCheckBox("複製圖片")
        self.autolabel_copy_images_check.setChecked(True)
        output_layout.addWidget(self.autolabel_copy_images_check)

        self.autolabel_output_yolo_check = QCheckBox("輸出 YOLO txt")
        self.autolabel_output_yolo_check.setChecked(False)
        output_layout.addWidget(self.autolabel_output_yolo_check)

        self.autolabel_output_json_check = QCheckBox("輸出 AnyLabel JSON")
        self.autolabel_output_json_check.setChecked(False)
        output_layout.addWidget(self.autolabel_output_json_check)

        self.autolabel_keep_conf_check = QCheckBox("YOLO txt 保留信心度")
        self.autolabel_keep_conf_check.setChecked(False)
        self.autolabel_keep_conf_check.setEnabled(False)
        
        # 連動 YOLO txt 勾選狀態與保留信心度的可用性
        self.autolabel_output_yolo_check.stateChanged.connect(
            lambda state: self.autolabel_keep_conf_check.setEnabled(state == Qt.CheckState.Checked.value)
        )
        output_layout.addWidget(self.autolabel_keep_conf_check)
        
        layout.addWidget(output_group)

        run_button = QPushButton("開始執行 Auto Label")
        run_button.clicked.connect(self.start_auto_label)
        self.register_run_button(run_button)
        layout.addWidget(run_button)
        layout.addStretch()
        return tab

    def add_path_row(
        self,
        parent_layout: QVBoxLayout,
        label: str,
        mode: str,
        file_filter: str = "All Files (*)",
    ) -> QLineEdit:
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        edit = QLineEdit()
        row.addWidget(edit)
        browse_button = QPushButton("瀏覽")
        browse_button.clicked.connect(
            lambda: self.browse_path(edit, mode=mode, file_filter=file_filter)
        )
        row.addWidget(browse_button)
        parent_layout.addLayout(row)
        return edit

    def register_run_button(self, button: QPushButton) -> None:
        button.setMinimumHeight(38)
        button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.run_buttons.append(button)

    def browse_path(self, target: QLineEdit, mode: str, file_filter: str = "All Files (*)") -> None:
        if mode == "dynamic":
            mode = "file" if self.test_source_combo.currentText() == "file" else "folder"

        if mode == "file":
            path, _ = QFileDialog.getOpenFileName(self, "選擇檔案", "", file_filter)
        else:
            path = QFileDialog.getExistingDirectory(self, "選擇資料夾")

        if path:
            target.setText(path)

    def setup_logging(self) -> None:
        self.log_handler = GUILogHandler()
        self.log_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        self.log_handler.log_signal.connect(self.append_log)

        root_logger = logging.getLogger()
        root_logger.addHandler(self.log_handler)
        root_logger.setLevel(logging.INFO)

    def append_log(self, message: str) -> None:
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def update_yolo_test_source_ui(self, source_type: str) -> None:
        is_youtube = source_type == "youtube"
        self.test_path_edit.setEnabled(not is_youtube)
        self.test_count_spin.setEnabled(is_youtube)

    def validate_required_paths(self, fields: list[tuple[QLineEdit, str]]) -> bool:
        for edit, name in fields:
            if not edit.text().strip():
                QMessageBox.warning(self, "警告", f"請先設定{name}！")
                return False
        return True

    def start_worker(self, task_name: str, task: Callable[[], object]) -> None:
        for button in self.run_buttons:
            button.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()

        self.worker = TaskWorker(task_name=task_name, task=task)
        self.worker.finished.connect(self.on_finished)
        self.worker.progress.connect(self.on_progress)
        self.worker.start()

    def start_deduplication(self) -> None:
        if not self.validate_required_paths(
            [
                (self.dedup_input_edit, "輸入圖片資料夾"),
                (self.dedup_output_edit, "輸出資料夾"),
            ]
        ):
            return
        if self.dedup_use_yolo_check.isChecked() and not self.dedup_yolo_edit.text().strip():
            QMessageBox.warning(self, "警告", "已啟用 YOLO 策略但未設定權重路徑！")
            return

        def task() -> list[str]:
            assert self.worker is not None
            service = DeduplicationService(
                threshold=float(self.dedup_threshold_spin.value()),
                yolo_weights=(
                    self.dedup_yolo_edit.text().strip()
                    if self.dedup_use_yolo_check.isChecked()
                    else None
                ),
            )
            return service.execute(
                input_folder=Path(self.dedup_input_edit.text().strip()),
                output_folder=Path(self.dedup_output_edit.text().strip()),
                use_confidence=self.dedup_use_yolo_check.isChecked(),
                sample_way=self.dedup_sample_way_combo.currentText(),
                write_mode=self.dedup_write_mode_combo.currentText(),
                progress_callback=self.worker.emit_progress,
            )

        self.start_worker("圖片去重", task)

    def start_negative_sampling(self) -> None:
        if not self.validate_required_paths(
            [
                (self.negative_input_edit, "輸入圖片資料夾"),
                (self.negative_output_edit, "輸出資料夾"),
                (self.negative_yolo_edit, "YOLO 權重路徑"),
            ]
        ):
            return

        def task() -> list[str]:
            service = NegativeSamplingService(
                yolo_weights=self.negative_yolo_edit.text().strip(),
                temperature=float(self.negative_temperature_spin.value()),
            )
            return service.execute(
                input_folder=Path(self.negative_input_edit.text().strip()),
                output_folder=Path(self.negative_output_edit.text().strip()),
                num_samples=int(self.negative_num_spin.value()),
            )

        self.start_worker("負樣本抽樣", task)

    def start_validation_clean(self) -> None:
        if not self.validate_required_paths(
            [
                (self.val_source_edit, "來源圖片資料夾"),
                (self.val_output_edit, "輸出 dataset 資料夾"),
                (self.val_yolo_edit, "YOLO 權重路徑"),
            ]
        ):
            return

        def task() -> None:
            service = ValidationCleanService(
                yolo_weights=self.val_yolo_edit.text().strip(),
                threshold=float(self.val_threshold_spin.value()),
            )
            service.execute(
                source_path=Path(self.val_source_edit.text().strip()),
                output_path=Path(self.val_output_edit.text().strip()),
            )

        self.start_worker("驗證集清洗", task)

    def start_yolo_test(self) -> None:
        source_type = self.test_source_combo.currentText()
        required_fields = [(self.test_yolo_edit, "YOLO 權重路徑")]
        if source_type != "youtube":
            required_fields.append((self.test_path_edit, "本地路徑"))
        if not self.validate_required_paths(required_fields):
            return

        def task() -> None:
            service = YoloTestService(
                yolo_weights=self.test_yolo_edit.text().strip(),
                conf=float(self.test_conf_spin.value()),
            )
            service.execute(
                source_type=source_type,
                path=(Path(self.test_path_edit.text().strip()) if source_type != "youtube" else None),
                count=int(self.test_count_spin.value()),
                output=(Path(self.test_output_edit.text().strip()) if self.test_output_edit.text().strip() else None),
            )

        self.start_worker("YOLO 測試", task)

    def start_auto_label(self) -> None:
        if not self.validate_required_paths(
            [
                (self.autolabel_input_edit, "輸入圖片資料夾"),
                (self.autolabel_output_edit, "輸出資料夾"),
                (self.autolabel_yolo_edit, "YOLO 權重路徑"),
            ]
        ):
            return
            
        if not (self.autolabel_copy_images_check.isChecked() or 
                self.autolabel_output_yolo_check.isChecked() or 
                self.autolabel_output_json_check.isChecked()):
            QMessageBox.warning(self, "警告", "至少需要選擇一種輸出（圖片、YOLO txt 或 AnyLabel JSON）！")
            return

        def task() -> list[object]:
            # 動態載入避免啟動過慢
            try:
                from .services import AutoLabelService
            except ImportError:
                from services import AutoLabelService
                
            service = AutoLabelService(
                yolo_weights=self.autolabel_yolo_edit.text().strip(),
                confidence_threshold=float(self.autolabel_conf_spin.value()),
                similarity_threshold=float(self.autolabel_sim_spin.value()),
            )
            return service.execute(
                input_folder=Path(self.autolabel_input_edit.text().strip()),
                output_folder=Path(self.autolabel_output_edit.text().strip()),
                copy_images=self.autolabel_copy_images_check.isChecked(),
                output_yolo_txt=self.autolabel_output_yolo_check.isChecked(),
                output_anylabel_json=self.autolabel_output_json_check.isChecked(),
                keep_confidence=self.autolabel_keep_conf_check.isChecked(),
                progress_callback=self.worker.emit_progress if self.worker else None,
            )

        self.start_worker("Auto Label", task)

    def on_progress(self, current: int, total: int) -> None:
        if total <= 0:
            self.progress_bar.setRange(0, 0)
            return
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(min(current, total))

    def on_finished(self, success: bool, message: str) -> None:
        for button in self.run_buttons:
            button.setEnabled(True)
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
