"""Tkinter GUI for the generic file compare/copy/delete utility.

The GUI uses natural-language workflows such as "以來源為基準清理目標"
instead of exposing internal enum names to end users.
"""

from __future__ import annotations

import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

try:
    from .file_compare_core import (
        CompareMode,
        FileCompareResult,
        GenericFileCompareEngine,
        Operation,
        PlannedAction,
        parse_patterns,
    )
except ImportError:
    from file_compare_core import (  # type: ignore[no-redef]
        CompareMode,
        FileCompareResult,
        GenericFileCompareEngine,
        Operation,
        PlannedAction,
        parse_patterns,
    )


@dataclass(frozen=True)
class ModeOption:
    value: CompareMode
    short_help: str
    sentence: str


@dataclass(frozen=True)
class OperationOption:
    value: Operation
    short_help: str
    sentence: str
    handles_sentence: str
    action_target: str
    condition: str
    no_touch: str
    needs_destination: bool = False
    is_delete: bool = False


MODE_OPTIONS: dict[str, ModeOption] = {
    "用相對路徑判斷是不是同一個檔案（最適合同步資料夾）": ModeOption(
        CompareMode.RELATIVE_PATH,
        "比較檔案在資料夾內的位置，例如 source/sub/a.txt 對 target/sub/a.txt。",
        "系統會把兩邊資料夾內的『相對路徑』拿來比較；路徑一樣就視為同一個檔案。",
    ),
    "只用完整檔名判斷是不是同一個檔案": ModeOption(
        CompareMode.FULL_NAME,
        "只看檔名與副檔名，例如 a.jpg 對 a.jpg，不管在哪個子資料夾。",
        "系統會忽略資料夾位置，只用完整檔名判斷兩邊是否有同一個檔案。",
    ),
    "只用主檔名判斷是不是同一組檔案（圖片 + OCR/JSON/TXT 推薦）": ModeOption(
        CompareMode.BASE_NAME,
        "只看副檔名前的名稱，例如 0001.jpg、0001.json、0001.txt 會被視為同一組。",
        "系統會用主檔名配對不同副檔名，適合檢查圖片與 OCR 結果、JSON、TXT 是否成對。",
    ),
    "用檔案內容判斷是否相同（最準但較慢，可用 C++ 加速）": ModeOption(
        CompareMode.CONTENT_HASH,
        "讀取檔案內容計算 hash；適合找內容一樣但檔名不同的檔案。",
        "系統會讀取檔案內容計算 hash；內容一樣才視為相同。大量檔案時可指定 C++ 加速器。",
    ),
}

OPERATION_OPTIONS: dict[str, OperationOption] = {
    "只看差異，不動檔案": OperationOption(
        Operation.REPORT_ONLY,
        "安全預覽：只產生統計與差異清單。",
        "只比較【來源資料夾】與【目標資料夾】的差異，不複製也不刪除任何檔案。",
        "這次不會處理任何檔案，只會列出差異。",
        "不複製、不刪除",
        "只列出來源與目標的差異",
        "不會改動來源或目標資料夾",
    ),
    "複製來源多出的檔案：複製「來源有、目標沒有」的檔案到輸出資料夾": OperationOption(
        Operation.COPY_SOURCE_ONLY,
        "把來源多出來、目標缺少的檔案複製到 Destination。",
        "找出【來源有、目標沒有】的檔案，並複製來源那一份到輸出資料夾。",
        "本次會處理：來源中那些目標不存在的檔案。",
        "複製來源檔案到輸出資料夾",
        "檔案在【來源】有，但在【目標】沒有",
        "不會刪除來源或目標資料夾裡的檔案",
        needs_destination=True,
    ),
    "複製目標多出的檔案：複製「目標有、來源沒有」的檔案到輸出資料夾": OperationOption(
        Operation.COPY_TARGET_ONLY,
        "把目標多出來、來源沒有的檔案複製到 Destination。",
        "找出【目標有、來源沒有】的檔案，並複製目標那一份到輸出資料夾。",
        "本次會處理：目標中那些來源不存在的檔案。",
        "複製目標檔案到輸出資料夾",
        "檔案在【目標】有，但在【來源】沒有",
        "不會刪除來源或目標資料夾裡的檔案",
        needs_destination=True,
    ),
    "抽出兩邊都有的檔案：複製『來源與目標都存在』的來源檔案": OperationOption(
        Operation.COPY_MATCHED_SOURCE,
        "把來源中與目標相符的檔案複製到 Destination。",
        "找出【來源】與【目標】都存在的檔案，並複製來源那一份到輸出資料夾。",
        "本次會處理：來源與目標都存在的檔案。",
        "複製來源檔案到輸出資料夾",
        "檔案在【來源】與【目標】都存在",
        "不會刪除來源或目標資料夾裡的檔案",
        needs_destination=True,
    ),
    "刪除來源多出的檔案：刪除「來源有、目標沒有」的檔案": OperationOption(
        Operation.DELETE_SOURCE_ONLY,
        "刪除來源端多出來、目標端不存在的檔案。",
        "刪除【來源資料夾】中那些【目標資料夾沒有】的檔案。",
        "本次會處理：來源中那些目標不存在的檔案。",
        "刪除來源檔案",
        "檔案在【來源】有，但在【目標】沒有",
        "不會刪除目標資料夾裡的檔案",
        is_delete=True,
    ),
    "刪除目標多出的檔案：刪除「目標有、來源沒有」的檔案": OperationOption(
        Operation.DELETE_TARGET_ONLY,
        "刪除目標端多出來、來源端不存在的檔案。你問的『刪除目標不存在於來源』就是選這個。",
        "刪除【目標資料夾】中那些【來源資料夾沒有】的檔案。",
        "本次會處理：目標中那些來源不存在的檔案。",
        "刪除目標檔案",
        "檔案在【目標】有，但在【來源】沒有",
        "不會刪除來源資料夾裡的檔案",
        is_delete=True,
    ),
    "清掉目標中已經和來源重複的檔案：刪除『目標與來源都存在』的目標檔案": OperationOption(
        Operation.DELETE_TARGET_MATCHES,
        "刪除目標中與來源相符的檔案。這比較危險，請確認用途。",
        "找出【來源】與【目標】都存在的檔案，並刪除目標那一份。",
        "本次會處理：來源與目標都存在的目標檔案。",
        "刪除目標檔案",
        "檔案在【來源】與【目標】都存在",
        "不會刪除來源資料夾裡的檔案",
        is_delete=True,
    ),
}

SCENARIO_OPTIONS: dict[str, tuple[str, str, bool]] = {
    "我想讓目標跟來源一樣乾淨": (
        "刪除目標多出的檔案：刪除「目標有、來源沒有」的檔案",
        "用相對路徑判斷是不是同一個檔案（最適合同步資料夾）",
        True,
    ),
    "我想找來源比目標多了哪些檔案": (
        "只看差異，不動檔案",
        "用相對路徑判斷是不是同一個檔案（最適合同步資料夾）",
        True,
    ),
    "我想把來源多出的檔案複製出來": (
        "複製來源多出的檔案：複製「來源有、目標沒有」的檔案到輸出資料夾",
        "用相對路徑判斷是不是同一個檔案（最適合同步資料夾）",
        True,
    ),
    "我想比對圖片和 OCR 結果是否成對": (
        "只看差異，不動檔案",
        "只用主檔名判斷是不是同一組檔案（圖片 + OCR/JSON/TXT 推薦）",
        True,
    ),
    "我想找內容一樣但檔名可能不同的檔案": (
        "只看差異，不動檔案",
        "用檔案內容判斷是否相同（最準但較慢，可用 C++ 加速）",
        True,
    ),
}


class FileCompareApp(tk.Tk):
    """Step-by-step GUI wrapper for :class:`GenericFileCompareEngine`."""

    def __init__(self) -> None:
        super().__init__()
        self.title("TVRA 通用檔案比對工具")
        self.geometry("1140x820")
        self.minsize(1040, 760)
        self._build_ui()
        self._refresh_help_text()

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill=tk.BOTH, expand=True)

        self.sources_var = tk.StringVar()
        self.target_var = tk.StringVar()
        self.destination_var = tk.StringVar()
        self.report_var = tk.StringVar()
        self.accelerator_var = tk.StringVar()
        self.include_var = tk.StringVar()
        self.exclude_var = tk.StringVar()
        self.mode_label_var = tk.StringVar(value=next(iter(MODE_OPTIONS)))
        self.operation_label_var = tk.StringVar(value=next(iter(OPERATION_OPTIONS)))
        self.scenario_label_var = tk.StringVar(value="請選一個常用情境（可不選）")
        self.dry_run_var = tk.BooleanVar(value=True)
        self.recursive_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="就緒：建議先保持 Dry Run，按『開始預覽 / 執行』查看計畫。")

        self.mode_label_var.trace_add("write", lambda *_: self._refresh_help_text())
        self.operation_label_var.trace_add("write", lambda *_: self._refresh_help_text())
        self.dry_run_var.trace_add("write", lambda *_: self._refresh_help_text())

        title = ttk.Label(root, text="TVRA 通用檔案比對 / 複製 / 刪除工具", font=("Microsoft JhengHei UI", 15, "bold"))
        title.pack(anchor=tk.W)
        subtitle = ttk.Label(
            root,
            text="用明確語句選擇『誰有、誰沒有、要動誰』。預設 Dry Run，只預覽、不改檔案。",
        )
        subtitle.pack(anchor=tk.W, pady=(2, 10))

        main = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main)
        right = ttk.Frame(main)
        main.add(left, weight=3)
        main.add(right, weight=2)

        self._build_scenarios(left)
        self._build_step_paths(left)
        self._build_step_action(left)
        self._build_step_compare(left)
        self._build_run_area(left)
        self._build_help_panel(right)
        self._build_output_panel(root)

    def _build_scenarios(self, parent: ttk.Frame) -> None:
        group = ttk.LabelFrame(parent, text="常用情境｜不確定要選什麼時，先從這裡開始")
        group.pack(fill=tk.X, pady=(0, 8))
        row = ttk.Frame(group)
        row.pack(fill=tk.X, pady=4)
        ttk.Label(row, text="我要做的是", width=18).pack(side=tk.LEFT)
        combo = ttk.Combobox(
            row,
            textvariable=self.scenario_label_var,
            values=["請選一個常用情境（可不選）", *SCENARIO_OPTIONS.keys()],
            state="readonly",
        )
        combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(row, text="套用情境", command=self._apply_scenario).pack(side=tk.RIGHT)

    def _build_step_paths(self, parent: ttk.Frame) -> None:
        group = ttk.LabelFrame(parent, text="Step 1｜選擇資料夾")
        group.pack(fill=tk.X, pady=(0, 8))
        self._path_row(group, "來源資料夾 Source", self.sources_var, self._browse_source, "你想拿來當參考或比對的一邊；可加入多個來源。")
        self._path_row(group, "目標資料夾 Target", self.target_var, lambda: self._browse_dir(self.target_var), "要被比對、被補齊或被清理的一邊。")
        self.destination_entry = self._path_row(
            group,
            "輸出資料夾 Destination",
            self.destination_var,
            lambda: self._browse_dir(self.destination_var),
            "只有『複製』操作需要；刪除或純報告可留空。",
        )
        self._path_row(group, "JSON 報告", self.report_var, self._browse_report, "可留空；建議保留報告方便回查完整清單。")

    def _build_step_action(self, parent: ttk.Frame) -> None:
        group = ttk.LabelFrame(parent, text="Step 2｜選擇你想做什麼")
        group.pack(fill=tk.X, pady=(0, 8))
        row = ttk.Frame(group)
        row.pack(fill=tk.X, pady=4)
        ttk.Label(row, text="操作目的", width=18).pack(side=tk.LEFT)
        self.operation_combo = ttk.Combobox(
            row,
            textvariable=self.operation_label_var,
            values=list(OPERATION_OPTIONS),
            state="readonly",
        )
        self.operation_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        options = ttk.Frame(group)
        options.pack(fill=tk.X, pady=4)
        ttk.Checkbutton(options, text="Dry Run：只預覽，不實際複製/刪除（建議先開啟）", variable=self.dry_run_var).pack(side=tk.LEFT)
        ttk.Checkbutton(options, text="遞迴掃描子資料夾", variable=self.recursive_var).pack(side=tk.LEFT, padx=14)

    def _build_step_compare(self, parent: ttk.Frame) -> None:
        group = ttk.LabelFrame(parent, text="Step 3｜選擇怎麼判斷『同一個檔案』")
        group.pack(fill=tk.X, pady=(0, 8))
        row = ttk.Frame(group)
        row.pack(fill=tk.X, pady=4)
        ttk.Label(row, text="比對方式", width=18).pack(side=tk.LEFT)
        self.mode_combo = ttk.Combobox(
            row,
            textvariable=self.mode_label_var,
            values=list(MODE_OPTIONS),
            state="readonly",
        )
        self.mode_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        self._path_row(
            group,
            "C++ Hash 加速器",
            self.accelerator_var,
            self._browse_accelerator,
            "只有『用檔案內容判斷』時才需要；可留空使用 Python fallback。",
        )

        filters = ttk.LabelFrame(group, text="檔案過濾（可留空，多個 glob 用 ; 分隔）")
        filters.pack(fill=tk.X, pady=6)
        self._entry_row(filters, "Include", self.include_var, "例如 *.jpg;*.json，只處理符合者。")
        self._entry_row(filters, "Exclude", self.exclude_var, "例如 *_backup.*;*.tmp，排除符合者。")

    def _build_run_area(self, parent: ttk.Frame) -> None:
        group = ttk.Frame(parent)
        group.pack(fill=tk.X, pady=(0, 8))
        buttons = ttk.Frame(group)
        buttons.pack(fill=tk.X)
        self.run_button = ttk.Button(buttons, text="開始預覽 / 執行", command=self._run_in_thread)
        self.run_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(buttons, text="重設所有欄位", command=self._reset_all).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(buttons, text="只清空結果", command=self._clear_output).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(group, textvariable=self.status_var, foreground="#555555").pack(anchor=tk.W, pady=(4, 0))

    def _build_help_panel(self, parent: ttk.Frame) -> None:
        help_group = ttk.LabelFrame(parent, text="目前設定說明｜用白話確認你現在要做什麼")
        help_group.pack(fill=tk.BOTH, expand=True, padx=(10, 0), pady=(0, 8))
        self.help_text = tk.Text(help_group, height=22, wrap=tk.WORD, padx=8, pady=8)
        self.help_text.pack(fill=tk.BOTH, expand=True)
        self.help_text.configure(state=tk.DISABLED)

        examples = ttk.LabelFrame(parent, text="例子：你問的操作要選哪個？")
        examples.pack(fill=tk.X, padx=(10, 0))
        ttk.Label(
            examples,
            text=(
                "想刪除『目標存在，但來源不存在』的檔案：\n"
                "→ 選「刪除目標多出的檔案」\n\n"
                "想補出來源比目標多的檔案：\n"
                "→ 選「複製來源多出的檔案」\n\n"
                "不確定前先保持 Dry Run，結果區會列出即將處理的檔案。"
            ),
            justify=tk.LEFT,
        ).pack(anchor=tk.W, padx=8, pady=8)

    def _build_output_panel(self, parent: ttk.Frame) -> None:
        output_group = ttk.LabelFrame(parent, text="結果摘要與預計動作（最多顯示前 50 筆）")
        output_group.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.output = scrolledtext.ScrolledText(output_group, height=16, wrap=tk.NONE)
        self.output.pack(fill=tk.BOTH, expand=True)

    def _path_row(self, parent: ttk.Frame, label: str, variable: tk.StringVar, command, hint: str) -> ttk.Entry:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text=label, width=18).pack(side=tk.LEFT)
        entry = ttk.Entry(row, textvariable=variable)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(row, text="瀏覽", command=command).pack(side=tk.RIGHT)
        ttk.Label(parent, text=f"  {hint}", foreground="#666666").pack(anchor=tk.W)
        return entry

    def _entry_row(self, parent: ttk.Frame, label: str, variable: tk.StringVar, hint: str) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text=label, width=18).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=variable).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Label(parent, text=f"  {hint}", foreground="#666666").pack(anchor=tk.W)

    def _browse_source(self) -> None:
        path = filedialog.askdirectory(title="選擇來源資料夾")
        if path:
            current = self.sources_var.get().strip()
            self.sources_var.set(f"{current};{path}" if current else path)

    @staticmethod
    def _set_if_selected(variable: tk.StringVar, value: str) -> None:
        if value:
            variable.set(value)

    def _browse_dir(self, variable: tk.StringVar) -> None:
        self._set_if_selected(variable, filedialog.askdirectory())

    def _browse_report(self) -> None:
        self._set_if_selected(
            variable=self.report_var,
            value=filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json"), ("All", "*.*")]),
        )

    def _browse_accelerator(self) -> None:
        self._set_if_selected(
            variable=self.accelerator_var,
            value=filedialog.askopenfilename(filetypes=[("Executable", "*.exe"), ("All", "*.*")]),
        )

    def _apply_scenario(self) -> None:
        scenario = self.scenario_label_var.get()
        if scenario not in SCENARIO_OPTIONS:
            return
        operation_label, mode_label, dry_run = SCENARIO_OPTIONS[scenario]
        self.operation_label_var.set(operation_label)
        self.mode_label_var.set(mode_label)
        self.dry_run_var.set(dry_run)
        self.status_var.set("已套用常用情境。請確認資料夾後先 Dry Run 預覽。")

    @staticmethod
    def _dry_run_sentence(dry_run: bool, operation_option: OperationOption) -> str:
        if dry_run:
            return "如果 Dry Run 開啟：只列出清單，不會複製或刪除。"
        if operation_option.is_delete:
            return "如果 Dry Run 關閉：會真的刪除這些檔案。"
        if operation_option.needs_destination:
            return "如果 Dry Run 關閉：會真的複製這些檔案到 Destination。"
        return "如果 Dry Run 關閉：仍只會列出差異，因為目前選的是純報告。"

    def _reset_all(self) -> None:
        self.sources_var.set("")
        self.target_var.set("")
        self.destination_var.set("")
        self.report_var.set("")
        self.accelerator_var.set("")
        self.include_var.set("")
        self.exclude_var.set("")
        self.mode_label_var.set(next(iter(MODE_OPTIONS)))
        self.operation_label_var.set(next(iter(OPERATION_OPTIONS)))
        self.scenario_label_var.set("請選一個常用情境（可不選）")
        self.dry_run_var.set(True)
        self.recursive_var.set(True)
        self._clear_output()
        self.status_var.set("已重設所有欄位。")

    def _clear_output(self) -> None:
        if hasattr(self, "output"):
            self.output.delete("1.0", tk.END)
        self.status_var.set("已清空結果。")

    def _refresh_help_text(self) -> None:
        operation_option = OPERATION_OPTIONS[self.operation_label_var.get()]
        mode_option = MODE_OPTIONS[self.mode_label_var.get()]
        dry_run = self.dry_run_var.get()

        if hasattr(self, "destination_entry"):
            state = tk.NORMAL if operation_option.needs_destination else tk.DISABLED
            self.destination_entry.configure(state=state)

        if operation_option.is_delete and not dry_run:
            warning = "⚠️ 你已關閉 Dry Run：按下執行後，符合條件的檔案會真的被刪除。"
        elif dry_run:
            warning = "✅ Dry Run 已開啟：這次只會列出清單，不會改動檔案。"
        else:
            warning = "⚠️ Dry Run 已關閉：複製操作會真的寫入輸出資料夾。"

        text = (
            "目前這個操作的意思\n"
            "=" * 34
            + "\n"
            f"{operation_option.sentence}\n"
            f"{operation_option.handles_sentence}\n"
            f"操作對象：{operation_option.action_target}\n"
            f"判斷條件：{operation_option.condition}\n"
            f"安全邊界：{operation_option.no_touch}\n"
            f"{self._dry_run_sentence(dry_run, operation_option)}\n\n"
            "怎麼判斷檔案是否存在於另一邊\n"
            "=" * 34
            + "\n"
            f"{mode_option.sentence}\n\n"
            f"目前安全狀態：{warning}\n\n"
            "自然語言對照\n"
            "=" * 34
            + "\n"
            "- 『來源有、目標沒有』：代表目標缺少來源中的這些檔案。\n"
            "- 『目標有、來源沒有』：代表目標多出來源中不存在的這些檔案。\n"
            "- 你要『刪除目標中來源不存在的檔案』，請選：刪除目標多出的檔案。\n"
        )
        if hasattr(self, "help_text"):
            self.help_text.configure(state=tk.NORMAL)
            self.help_text.delete("1.0", tk.END)
            self.help_text.insert(tk.END, text)
            self.help_text.configure(state=tk.DISABLED)

    def _run_in_thread(self) -> None:
        try:
            self._validate_before_run()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("設定不完整", str(exc))
            return

        operation_option = OPERATION_OPTIONS[self.operation_label_var.get()]
        if operation_option.is_delete and not self.dry_run_var.get():
            confirmed = messagebox.askyesno(
                "確認刪除",
                f"{operation_option.sentence}\n\n"
                "你已關閉 Dry Run，接下來會實際刪除檔案。\n"
                "建議先開啟 Dry Run 檢查 planned actions。\n\n"
                "確定要繼續嗎？",
            )
            if not confirmed:
                return

        self.run_button.configure(state=tk.DISABLED)
        self.status_var.set("執行中，請稍候...")
        threading.Thread(target=self._run, daemon=True).start()

    def _validate_before_run(self) -> None:
        sources = [part.strip() for part in self.sources_var.get().split(";") if part.strip()]
        if not sources:
            raise ValueError("請至少選擇一個來源資料夾。")
        if not self.target_var.get().strip():
            raise ValueError("請選擇目標資料夾。")
        operation_option = OPERATION_OPTIONS[self.operation_label_var.get()]
        if operation_option.needs_destination and not self.destination_var.get().strip():
            raise ValueError("目前選擇的是複製操作，請設定輸出資料夾 Destination。")

    def _run(self) -> None:
        try:
            sources = [Path(part.strip()) for part in self.sources_var.get().split(";") if part.strip()]
            mode_option = MODE_OPTIONS[self.mode_label_var.get()]
            operation_option = OPERATION_OPTIONS[self.operation_label_var.get()]
            engine = GenericFileCompareEngine(Path(self.accelerator_var.get()) if self.accelerator_var.get().strip() else None)
            result = engine.run(
                source_directories=sources,
                target_directory=Path(self.target_var.get().strip()),
                compare_mode=mode_option.value,
                operation=operation_option.value,
                destination_directory=Path(self.destination_var.get().strip()) if self.destination_var.get().strip() else None,
                dry_run=self.dry_run_var.get(),
                recursive=self.recursive_var.get(),
                include_patterns=parse_patterns(self.include_var.get()),
                exclude_patterns=parse_patterns(self.exclude_var.get()),
                report_path=Path(self.report_var.get().strip()) if self.report_var.get().strip() else None,
            )
            self.after(0, lambda: self._show_result(result))
        except Exception as exc:  # noqa: BLE001
            self.after(0, lambda: self._show_error(exc))

    def _show_result(self, result: FileCompareResult) -> None:
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, self._format_result(result))
        self.run_button.configure(state=tk.NORMAL)
        action = "預覽" if result.dry_run else "執行"
        self.status_var.set(f"{action}完成：規劃 {len(result.planned_actions)} 筆動作，實際執行 {len(result.executed_actions)} 筆。")

    def _show_error(self, exc: Exception) -> None:
        self.run_button.configure(state=tk.NORMAL)
        self.status_var.set("執行失敗。")
        messagebox.showerror("執行失敗", str(exc))

    def _format_result(self, result: FileCompareResult) -> str:
        operation_option = OPERATION_OPTIONS[self.operation_label_var.get()]
        source_only_count = len(result.source_only_files)
        target_only_count = len(result.target_only_files)
        matched_count = len(result.matched_source_files)

        lines = [
            "結果摘要",
            "=" * 40,
            f"來源檔案總數：{result.source_file_count}",
            f"目標檔案總數：{result.target_file_count}",
            f"以【來源】看【目標】：目標缺少 {source_only_count} 個來源檔案",
            f"以【目標】看【來源】：目標多出 {target_only_count} 個來源沒有的檔案",
            f"來源與目標都存在：{matched_count} 個來源檔案可配對",
            f"比對方式：{result.compare_mode}",
            f"Dry Run：{'開啟，只預覽不改檔案' if result.dry_run else '關閉，已實際執行允許的動作'}",
            f"C++ 加速器：{'已使用' if result.accelerator_used else '未使用'}",
            "",
            "本次選擇的操作安全摘要",
            "=" * 40,
            f"操作對象：{operation_option.action_target}",
            f"判斷條件：{operation_option.condition}",
            f"安全邊界：{operation_option.no_touch}",
            "",
            operation_option.handles_sentence,
            self._operation_count_sentence(result),
            f"規劃動作：{len(result.planned_actions)} 筆",
            f"實際執行：{len(result.executed_actions)} 筆",
        ]
        if result.destination_directory:
            lines.append(f"輸出資料夾：{result.destination_directory}")

        lines.extend([
            "",
            "預計動作（最多顯示前 50 筆）",
            "=" * 40,
        ])
        if not result.planned_actions:
            lines.append("沒有需要複製或刪除的動作。")
        else:
            for index, action in enumerate(result.planned_actions[:50], start=1):
                if action.operation == "copy":
                    lines.append(f"{index}. 複製｜原因：{self._reason_label(action.reason)}")
                    lines.append(f"   來源：{action.source}")
                    lines.append(f"   目的：{action.destination}")
                else:
                    lines.append(f"{index}. 刪除｜原因：{self._reason_label(action.reason)}")
                    lines.append(f"   檔案：{action.source}")
            if len(result.planned_actions) > 50:
                lines.append(f"... 尚有 {len(result.planned_actions) - 50} 筆未顯示，請輸出 JSON 報告查看完整清單。")
        return "\n".join(lines)

    @staticmethod
    def _operation_count_sentence(result: FileCompareResult) -> str:
        if result.operation == Operation.DELETE_TARGET_ONLY.value:
            return f"本次將刪除：目標檔案；條件：目標有、來源沒有；數量：{len(result.target_only_files)} 個"
        if result.operation == Operation.DELETE_SOURCE_ONLY.value:
            return f"本次將刪除：來源檔案；條件：來源有、目標沒有；數量：{len(result.source_only_files)} 個"
        if result.operation == Operation.COPY_SOURCE_ONLY.value:
            return f"本次將複製：來源檔案；條件：來源有、目標沒有；數量：{len(result.source_only_files)} 個"
        if result.operation == Operation.COPY_TARGET_ONLY.value:
            return f"本次將複製：目標檔案；條件：目標有、來源沒有；數量：{len(result.target_only_files)} 個"
        if result.operation == Operation.COPY_MATCHED_SOURCE.value:
            return f"本次操作會處理：來源與目標都存在的 {len(result.matched_source_files)} 個檔案"
        if result.operation == Operation.DELETE_TARGET_MATCHES.value:
            return f"本次操作會處理：目標中也存在於來源的 {len(result.matched_target_files)} 個檔案"
        return "本次操作只看差異，不會處理任何檔案"

    @staticmethod
    def _reason_label(reason: str) -> str:
        labels = {
            "source-only": "來源存在，但目標不存在",
            "target-only": "目標存在，但來源不存在",
            "matched-source": "來源與目標都存在，複製來源檔案",
            "target-match": "目標與來源都存在",
        }
        return labels.get(reason, reason)


def main() -> None:
    app = FileCompareApp()
    app.mainloop()


if __name__ == "__main__":
    main()