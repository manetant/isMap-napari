# src/tcell_analysis/_widget.py
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import os
import pandas as pd
import json

from magicgui import magicgui
from magicgui.widgets import Container, PushButton, FileEdit, Label, ComboBox, SpinBox

#from PyQt5 import QtCore
#QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts, True)

from napari.qt.threading import thread_worker
from napari.utils import progress
from napari import current_viewer
from qtpy.QtWidgets import (
    QInputDialog, QApplication, QDialog, QListWidget, QListWidgetItem, QVBoxLayout,
    QPushButton, QTableWidget, QTableWidgetItem, QCheckBox, QLineEdit, QRadioButton,
    QDialogButtonBox, QWidget, QHBoxLayout, QButtonGroup
)
from bioio import BioImage
import tifffile as tiff
from qtpy.QtCore import Qt, QObject, Signal
from qtpy.QtWidgets import QProgressBar, QTextEdit, QGridLayout, QLabel

from .analysis import run_analysis
from .visualization.view_in_napari import show_analysis_results
from napari import current_viewer
from .visualization.view_in_napari import reset_tcell_session  # make sure this import is present
from qtpy.QtWidgets import QLineEdit, QMessageBox, QFileDialog
import time
from qtpy.QtCore import QTimer
from qtpy.QtGui import QPixmap
from qtpy.QtCore import Qt

from .metadata_utils import get_pixel_sizes_and_units, avg_px_um

def wrap_form_with_logo(base_form_native, logo_path: Path, width_px: int = 600) -> QWidget:
    """
    Returns a QWidget that shows a centered logo on top and your form under it.
    Does NOT dock anything (napari will dock the returned widget).
    """
    host = QWidget()
    lay = QVBoxLayout(host)
    lay.setContentsMargins(6, 6, 6, 6)
    lay.setSpacing(8)

    logo_lbl = QLabel()
    logo_lbl.setAlignment(Qt.AlignCenter)
    logo_lbl.setStyleSheet("background: transparent;")
    pix = QPixmap(str(logo_path))
    if not pix.isNull():
        logo_lbl.setPixmap(pix.scaledToWidth(width_px, Qt.SmoothTransformation))
    else:
        logo_lbl.setText("⚠️ Logo not found")

    lay.addWidget(logo_lbl)
    lay.addWidget(base_form_native)

    # give the dock a comfy default size
    host.setMinimumWidth(600)
    host.setMinimumHeight(222)
    return host

def _find_sample_image(base: Path) -> Path | None:
    # search inside base for a single image we can open to detect channels
    exts = (".vsi", ".nd2", ".czi", ".lif", ".tif", ".tiff")
    for ext in exts:
        hits = list(base.rglob(f"*{ext}"))
        if hits:
            return hits[0]
    return None

def make_dialog(title: str, parent=None, w: int = 600, h: int = 400) -> QDialog:
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.setMinimumSize(w, h)
    lay = QVBoxLayout(dlg)
    return dlg, lay

def _choose_channels_and_seg_pixel(parent, file_path: Path, prefer_name: str = "Actin"):
    """
    Pop a table listing detected channels:
      (selected_names_ordered, channel_map_orig_to_new, segmentation_name) 
    or (None, None, None) if cancelled.
    """
    # detect channels with BioImage
    img = BioImage(str(file_path))
    px_meta = get_pixel_sizes_and_units(img)
    um_per_px = avg_px_um(px_meta)

    detected = list(getattr(img, "channel_names", []) or [])
    if not detected:
        # fallback to OME-XML or generic C0.. if needed
        with tiff.TiffFile(str(file_path)) as tf:
            chs = tf.pages[0].tags.get("ImageDescription", None)
        # just make something reasonable if none
        detected = [f"C{i}" for i in range(int(img.shape["C"]))]

    dlg, lay = make_dialog("Select & Rename Channels, Pick Segmentation Channel", 
                           parent)


    table = QTableWidget(dlg)
    table.setColumnCount(4)
    table.setHorizontalHeaderLabels(["Keep", "Original", "Rename to", "Segment?"])
    table.setRowCount(len(detected))

    seg_group = QButtonGroup(table)
    seg_group.setExclusive(True)

    # defaults
    # keep all; rename = detected; segment = prefer_name if present else last
    default_seg_idx = next((i for i, nm in enumerate(detected) if nm.lower() == prefer_name.lower()), len(detected)-1)

    for r, orig in enumerate(detected):
        # Keep checkbox
        keep = QCheckBox()
        keep.setChecked(True)
        table.setCellWidget(r, 0, keep)

        # Original (read-only)
        it = QTableWidgetItem(orig)
        it.setFlags(it.flags() & ~Qt.ItemIsEditable)
        table.setItem(r, 1, it)

        # Rename to
        name_edit = QLineEdit(orig)
        table.setCellWidget(r, 2, name_edit)

        # Segment radio
        rb = QRadioButton()
        rb.setChecked(r == default_seg_idx)
        seg_group.addButton(rb, r)
        table.setCellWidget(r, 3, rb)

    table.resizeColumnsToContents()
    lay.addWidget(table)

    btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dlg)
    lay.addWidget(btns)
    btns.accepted.connect(dlg.accept)
    btns.rejected.connect(dlg.reject)

    if dlg.exec_() != QDialog.Accepted:
        return None, None, None, None

    # collect
    keep_rows = []
    for r in range(table.rowCount()):
        keep = table.cellWidget(r, 0).isChecked()
        orig = table.item(r, 1).text()
        new  = table.cellWidget(r, 2).text().strip() or orig
        if keep:
            keep_rows.append((orig, new, r))

    if not keep_rows:
        return None, None, None, None

    seg_row = seg_group.checkedId()
    seg_orig = detected[seg_row] if 0 <= seg_row < len(detected) else keep_rows[-1][0]
    # If segmentation channel wasn’t kept, force it to the first kept row
    if not any(orig == seg_orig for (orig, _, _) in keep_rows):
        seg_orig = keep_rows[0][0]

    selected_names = [new for (_, new, _) in keep_rows]
    channel_map = {orig: new for (orig, new, _) in keep_rows}

    # final segmentation name is the renamed one (if kept)
    seg_new = channel_map.get(seg_orig, keep_rows[0][1])
    return selected_names, channel_map, seg_new, um_per_px


@dataclass
class RowState:
    tag: str = ""
    tagged_path: str = ""
    conditions_by_subfolder: Dict[str, str] = field(default_factory=dict)
    conditions_for_path: str = ""

def _scan_done_tags(out_dir: Path) -> Dict[str, Path]:
    done: Dict[str, Path] = {}
    try:
        for csv_path in out_dir.rglob("per_cell_features.csv"):
            case_dir = csv_path.parent
            mask_candidates = list(case_dir.glob("*_mask.tiff"))
            if not mask_candidates:
                continue
            mask_path = mask_candidates[0]    # <-- FIX
            if not mask_path.exists():
                continue
            try:
                df = pd.read_csv(csv_path, nrows=1)
            except Exception:
                continue
            if len(df) == 0 or "tag" not in df.columns:
                continue
            tag = str(df["tag"].iloc[0])
            done[tag] = case_dir
    except Exception:
        pass
    return done

def _iter_immediate_subfolders(base: Path) -> List[Path]:
    if not base or not base.exists() or not base.is_dir():
        return []
    try:
        return [p for p in base.iterdir() if p.is_dir()]
    except Exception:
        return []

def _prompt_text(parent, title: str, label: str, default: str) -> str | None:
    """Prompt user for a text input, centered and wider than default."""
    dlg_parent = parent.native if hasattr(parent, "native") else parent

    dlg = QInputDialog(dlg_parent)
    dlg.setWindowTitle(title)
    dlg.setLabelText(label)
    dlg.setTextValue(default)
    dlg.setMinimumWidth(400)   # make wider
    dlg.resize(400, 150)       # optional: fix height too
    dlg.setSizeGripEnabled(True)

    # --- Center on screen ---
    screen = QApplication.primaryScreen().geometry()
    dlg_rect = dlg.frameGeometry()
    center_point = screen.center() - dlg_rect.center()
    dlg.move(center_point)

    if dlg.exec_() == QInputDialog.Accepted:
        txt = dlg.textValue().strip()
        return txt if txt else None
    return None


def _ask_conditions_for_subfolders(parent_widget, base_path: Path,
                                   existing: Dict[str, str] | None = None) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    subfolders = _iter_immediate_subfolders(base_path)
    for sub in sorted(subfolders):
        key = sub.name
        default = (existing or {}).get(key, key)
        val = _prompt_text(parent_widget, "Assign condition", f"Enter condition for:\n{key}", default)
        mapping[key] = val if val is not None else default
    return mapping

def _choose_conditions_dialog(parent, pairs: List[Tuple[Path, str]]) -> List[Tuple[Path, str]]:
    """
    Show a checkbox list of (path, label) and return the checked subset.
    'label' should be the condition/tag shown to user.
    """
    dlg = QDialog(parent)
    dlg.setWindowTitle("Select condition to set segmentation parameters")
    lay = QVBoxLayout(dlg)
    dlg.setMinimumSize(600, 400)
    lst = QListWidget(dlg)
    lst.setSelectionMode(QListWidget.NoSelection)
    for p, label in pairs:
        item = QListWidgetItem(f"{label}  —  {p.name}")
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)   # preselect all
        item.setData(Qt.UserRole, (str(p), label))  # use UserRole
        lst.addItem(item)
    lay.addWidget(lst)
    btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dlg)
    lay.addWidget(btns)
    btns.accepted.connect(dlg.accept)
    btns.rejected.connect(dlg.reject)
    if dlg.exec_() != QDialog.Accepted:
        return []
    out: List[Tuple[Path, str]] = []
    for i in range(lst.count()):
        it = lst.item(i)
        if it.checkState():
            p_str, label = it.data(Qt.UserRole)
            out.append((Path(p_str), label))
    return out


class _ProgressBridge(QObject):
    # Thread-safe signals
    total_changed = Signal(int, int)              # done, total
    file_changed = Signal(int, int)               # done, total for current file
    now_processing = Signal(str, str)             # path, tag
    log_line = Signal(str)
    finished = Signal()

class ProgressPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QGridLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        # Existing widgets
        self.lblOverall = QLabel("Overall: 0 / 0")
        self.pbOverall = QProgressBar()
        self.pbOverall.setMinimum(0); self.pbOverall.setMaximum(100)
        self.lblCurrent = QLabel("Current: –")
        self.pbCurrent = QProgressBar()
        self.pbCurrent.setMinimum(0); self.pbCurrent.setMaximum(100)
        self.lblTag = QLabel("Tag: –")
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setMinimumHeight(160)

        # NEW: elapsed time labels
        self.lblElapsedOverall = QLabel("Elapsed (overall): 00:00:00")

        # Layout
        lay.addWidget(self.lblOverall,         0, 0, 1, 2)
        lay.addWidget(self.pbOverall,          1, 0, 1, 2)
        lay.addWidget(self.lblCurrent,         2, 0, 1, 2)
        lay.addWidget(self.pbCurrent,          3, 0, 1, 2)
        lay.addWidget(self.lblTag,             4, 0, 1, 2)
        lay.addWidget(self.lblElapsedOverall,  5, 0, 1, 2)
        lay.addWidget(self.log,                7, 0, 1, 2)

        # Timing state
        self._overall_start = None   # type: float | None
        self._current_start = None 
        self._running = False

        # Tick every second to refresh labels
        self._tick = QTimer(self)
        self._tick.timeout.connect(self._on_tick)
        self._tick.start(1000)

    # ---- public API used by your bridge callbacks ----
    def set_total(self, done, total):
        # when a batch starts (done==0), start overall timer
        if total > 0:
            pct = 0 if total == 0 else int(100 * done / total)
            self.pbOverall.setValue(pct)
            self.lblOverall.setText(f"Overall: {done} / {total}")
        if done == 0 and total > 0 and not self._running:
            self._overall_start = time.monotonic()
            self._running = True
            self._update_elapsed_labels(force=True)
        if total > 0 and done >= total:
            # batch finished
            self._running = False
            self._update_elapsed_labels(force=True)

    def set_file(self, done, total):
        pct = 0 if total == 0 else int(100 * done / max(total, 1))
        self.pbCurrent.setValue(pct)
        # keep current elapsed updating via timer

    def set_now(self, path, tag):
        # called at the start of each case → reset "current" timer
        self.lblCurrent.setText(f"Current: {Path(path).name if path else '–'}")
        self.lblTag.setText(f"Tag: {tag if tag else '–'}")
        self._current_start = time.monotonic()
        self._update_elapsed_labels(force=True)

    def append(self, msg):
        cursor = self.log.textCursor()
        cursor.movePosition(cursor.Start)
        cursor.insertText("⏳ " + msg + "\n\n")
        self.log.setTextCursor(cursor)
        self.log.ensureCursorVisible()

    def mark_finished(self):
        self._running = False
        self._update_elapsed_labels(force=True)

    # ---- helpers ----
    def _format_hms(self, seconds: float) -> str:
        seconds = max(0, int(seconds))
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _update_elapsed_labels(self, force: bool = False):
        now = time.monotonic()
        # overall
        if self._overall_start is not None:
            overall_sec = (now - self._overall_start) if self._running else (self._last_overall() or 0)
            self.lblElapsedOverall.setText(f"Elapsed (overall): {self._format_hms(overall_sec)}")
        else:
            self.lblElapsedOverall.setText("Elapsed (overall): 00:00:00")

    def _on_tick(self):
        # update every second while running
        if self._running:
            self._update_elapsed_labels()

    def _last_overall(self):
        if self._overall_start is None:
            return 0
        return time.monotonic() - self._overall_start

    def _last_current(self):
        if self._current_start is None:
            return 0
        return time.monotonic() - self._current_start


def tcell_widget():
    # ---------- STATE ----------
    primary = RowState()
    extra_rows: List[FileEdit] = []
    state_by_row: Dict[FileEdit, RowState] = {}
    DEFAULT_WORKERS = 1
    # store filter thresholds picked during “Run Segmentation”
    filter_ranges: Dict[str, tuple] = {}
    chosen_channel_names: List[str] = []
    chosen_seg_channel: str | None = None
    chosen_channel_map: Dict[str, str] = {}
    um_per_px_global: float | None = None

    def _ensure_channel_selection(base_path_for_sampling: Path, parent_window) -> bool:
        nonlocal chosen_channel_names, chosen_seg_channel, chosen_channel_map, um_per_px_global

        if chosen_channel_names and chosen_seg_channel:
            return True

        sample = _find_sample_image(base_path_for_sampling)
        if sample is None:
            return False

        sel_names, ch_map, seg_new, um_per_px = _choose_channels_and_seg_pixel(parent_window, sample, prefer_name="Actin")
        if not sel_names:
            return False

        chosen_channel_names = sel_names
        chosen_seg_channel   = seg_new
        chosen_channel_map   = ch_map
        um_per_px_global     = um_per_px 
        return True

    # ---------- BASE FORM ----------
    @magicgui(
        input_folder={"widget_type": "FileEdit", "mode": "d", "label": "Input Folder"},
        output_folder={"widget_type": "FileEdit", "mode": "d", "label": "Output Folder"},
        #num_workers={"widget_type": "SpinBox", "min": 1, "max": 32, "value": 1},
        #save_extracted={"widget_type": "CheckBox", "label": "Save extracted cell crops", "value": True},
        
    )

    def form(
        input_folder: Path,
        output_folder: Path,
        #num_workers: int,
        #save_extracted: bool,
        seg_model: str,
        seg_diameter: int,
        seg_diam_units: str,
        seg_scale: float,        
    ):
        ...

    # ---- Segmentation Parameters Section ----
    seg_header = Label(value="<b>Segmentation Parameters</b> "
                             "<span style='color:#777'>(used only for segmentation)</span>")
    seg_model = ComboBox(label="Model", choices=["cyto3", "cyto2"], value="cyto3")
    seg_diameter = SpinBox(label="Diameter", min=0, max=500, value=100, step=1)
    seg_diam_units = ComboBox(label="Diameter units", choices=["px", "µm"], value="px")
    seg_scale = ComboBox(
        label="Scale (×)",
        choices=[f"{x/10:.1f}" for x in range(10, 2, -1)],
        value="0.7",
        tooltip="Scale applied to segmentation (1.0 = native size)",
    )

    seg_block = Container(layout="vertical", labels=True)
    seg_block.extend([seg_header, seg_model, seg_diameter, seg_diam_units, seg_scale])

    @form.input_folder.changed.connect
    def _on_primary_changed(path: Path | None):
        if not path:
            return
        cur = str(path)
        if not form.output_folder.value:
            form.output_folder.value = Path(cur) / "tcell-results"

        subs = _iter_immediate_subfolders(Path(cur))
        if subs:
            if primary.conditions_for_path != cur or not primary.conditions_by_subfolder:
                primary.conditions_by_subfolder = _ask_conditions_for_subfolders(form, Path(cur), primary.conditions_by_subfolder if primary.conditions_for_path == cur else None)
                primary.conditions_for_path = cur
            primary.tag = ""; primary.tagged_path = ""
        else:
            if primary.tagged_path != cur:
                default = Path(cur).name
                t = _prompt_text(form, "Tag this input", f"Enter a tag for “{default}”:", default)
                primary.tag = t or ""
                primary.tagged_path = cur
            primary.conditions_by_subfolder.clear(); primary.conditions_for_path = ""

    # ---------- BUILD UI ORDER ----------
    run_seg_btn = PushButton(label="Run Segmentation")
    run_all_btn = PushButton(label="Run Analysis")
    load_btn = PushButton(label="Load Data")
    reset_btn   = PushButton(label="Reset Session")

    # Make it look dangerous and distinct
    try:
        _qbtn = reset_btn.native  # this is a Qt QPushButton
        _qbtn.setToolTip("Reset everything in the viewer and widget state.")
        _qbtn.setMinimumWidth(160)
        _qbtn.setStyleSheet("""
            QPushButton {
                background: #fff5f5;
                color: #b00020;
                border: 1px solid #b00020;
                border-radius: 8px;
                padding: 6px 10px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #ffecec;
            }
            QPushButton:pressed {
                background: #ffdede;
            }
            QPushButton:focus {
                outline: none;
                border: 2px solid #b00020;
            }
        """)
    except Exception:
        pass

    ui = Container(layout="vertical", labels=True)
    ui.append(form.input_folder)
    #ui.append(add_btn)
    #ui.append(extra_box)
    #ui.append(form.num_workers)
    #ui.append(form.save_extracted)
    ui.append(form.output_folder)
    ui.append(seg_block)
    ui.append(run_seg_btn) 
    ui.append(run_all_btn)
    ui.append(load_btn) 
    ui.append(reset_btn)


    # ---- Progress dock (one per widget) ----
    progress_panel = ProgressPanel()
    bridge = _ProgressBridge()

    # connect signals to the panel (main thread safe)
    bridge.total_changed.connect(progress_panel.set_total)
    bridge.file_changed.connect(progress_panel.set_file)
    bridge.now_processing.connect(progress_panel.set_now)
    bridge.log_line.connect(progress_panel.append)
    
    bridge.finished.connect(progress_panel.mark_finished)

    # mount the dock (left or right as you prefer)
    try:
        current_viewer().window.add_dock_widget(progress_panel, area="left", name="isMap Progress")
    except Exception:
        pass

    def _clear_fileedit(fe):
        """Hard-clear a magicgui FileEdit: value + visible text, safely."""
        try:
            # stop event cascades (e.g., your _on_primary_changed auto-fill)
            fe.native.blockSignals(True)
        except Exception:
            pass

        try:
            # clear value first
            fe.value = None
        except Exception:
            pass

        # clear any line edits inside (covers .line_edit, ._line_edit, etc.)
        try:
            for le in fe.native.findChildren(QLineEdit):
                le.clear()
        except Exception:
            pass

        # best-effort on common attribute
        try:
            if hasattr(fe, "line_edit") and fe.line_edit is not None:
                fe.line_edit.setText("")
        except Exception:
            pass

        try:
            fe.native.blockSignals(False)
        except Exception:
            pass


    @reset_btn.changed.connect
    def _reset_session(_=None):
        viewer = current_viewer()
        if viewer is None:
            return

        # (optional) confirmation dialog here if you added it

        # --- temporarily disconnect the auto-fill so it won't repopulate output_folder
        disconnected = False
        try:
            form.input_folder.changed.disconnect(_on_primary_changed)
            disconnected = True
        except Exception:
            pass

        # --- deep clean napari (keep progress dock as discussed)
        try:
            reset_tcell_session(viewer)  # or reset_tcell_session(viewer, remove_progress=False)
        except Exception:
            pass

        # --- clear widget state
        filter_ranges.clear()
        chosen_channel_names.clear()
        nonlocal chosen_seg_channel; chosen_seg_channel = None
        chosen_channel_map.clear()
        primary.tag = ""; primary.tagged_path = ""
        primary.conditions_by_subfolder.clear(); primary.conditions_for_path = ""

        # --- clear file pickers (both value & visible text), with signals blocked
        try:
            _clear_fileedit(form.input_folder)
        except Exception:
            pass
        try:
            _clear_fileedit(form.output_folder)
        except Exception:
            pass

        # --- reset segmentation parameter widgets
        try:
            seg_model.value = "cyto3"
            seg_diameter.value = 100
            seg_diam_units.value = "px"
            seg_scale.value = "0.7"
        except Exception:
            pass

        # --- reconnect the auto-fill handler
        if disconnected:
            try:
                form.input_folder.changed.connect(_on_primary_changed)
            except Exception:
                pass

        # --- clear progress panel but keep it visible
        try:
            progress_panel.set_total(0, 0)
            progress_panel.set_file(0, 0)
            progress_panel.set_now("", "")
            progress_panel.log.clear()
        except Exception:
            pass

        try:
            viewer.status = "🔄 Session reset. Input/Output folders cleared."
        except Exception:
            pass

    def _gather_all_cases() -> List[Tuple[Path, str]]:
        tasks: List[Tuple[Path, str]] = []
        if form.input_folder.value:
            base = Path(form.input_folder.value)
            subs = _iter_immediate_subfolders(base)
            if subs:
                if not primary.conditions_by_subfolder:
                    primary.conditions_by_subfolder = _ask_conditions_for_subfolders(form, base, None)
                    primary.conditions_for_path = str(base)
                for s in sorted(subs):
                    tasks.append((s, primary.conditions_by_subfolder.get(s.name, s.name)))
            else:
                if not primary.tag.strip():
                    t = _prompt_text(form, "Tag this input", f"Enter a tag for “{base.name}”:", base.name)
                    if t:
                        primary.tag = t; primary.tagged_path = str(base)
                if primary.tag.strip():
                    tasks.append((base, primary.tag))
        return tasks
    
    # ---------- LOAD EXISTING ANALYSIS ----------
    @load_btn.changed.connect
    def _load_data(_=None):
        viewer = current_viewer()
        if viewer is None:
            return

        parent = getattr(viewer, "window", None)
        parent = getattr(parent, "_qt_window", None)

        # Ask user for the result folder
        # Use current output_folder (if set) as starting point
        start_dir = str(form.output_folder.value) if form.output_folder.value else str(Path.home())

        folder = QFileDialog.getExistingDirectory(
            parent,
            "Select analysis result folder",
            start_dir,
        )

        # User cancelled
        if not folder:
            try:
                viewer.status = "ℹ️ Load data cancelled."
            except Exception:
                pass
            return

        out_dir = Path(folder)
        # keep the form in sync with what the user picked
        form.output_folder.value = out_dir

        # Validate that this looks like an analysis output folder
        run_json = out_dir / "run.json"
        if not run_json.exists():
            if parent is not None:
                QMessageBox.warning(
                    parent,
                    "Load data",
                    f"No 'run.json' found in:\n{out_dir}\n\n"
                    "Please select the root output folder used when you ran the analysis.",
                )
            try:
                viewer.status = "⚠️ Selected folder does not look like an analysis output folder."
            except Exception:
                pass
            return

        # Read channel names from run.json
        try:
            meta = json.loads(run_json.read_text())
            chan_list = meta.get("channels") or []
        except Exception as e:
            if parent is not None:
                QMessageBox.critical(
                    parent,
                    "Load data",
                    f"Failed to read 'run.json':\n{e}",
                )
            try:
                viewer.status = "⚠️ Could not read run.json."
            except Exception:
                pass
            return

        if not chan_list:
            if parent is not None:
                QMessageBox.warning(
                    parent,
                    "Load data",
                    "The 'run.json' file does not contain a 'channels' list.",
                )
            try:
                viewer.status = "⚠️ No channel information in run.json."
            except Exception:
                pass
            return

        # Check that there is at least one mask to visualize
        mask_hits = list(out_dir.rglob("*_mask.tiff"))
        if not mask_hits:
            if parent is not None:
                QMessageBox.warning(
                    parent,
                    "Load data",
                    "No '*_mask.tiff' files found under this folder; nothing to visualize.",
                )
            try:
                viewer.status = "⚠️ No masks found in this folder."
            except Exception:
                pass
            return

        # Load into isMap visualization ---
        try:
            viewer.status = "📂 Loading existing analysis..."
        except Exception:
            pass

        try:
            show_analysis_results(
                viewer,
                out_dir,
                chan_list,
                tasks_with_tags=[],      # currently unused by show_analysis_results
                initial_ranges={},
                on_filter_change=None,
                show_filter=False,       # behave like 'Run Analysis' (full QC, no filter sliders)
                show_boxplot=True,
                show_radial_viewer=True,
                show_pcc=True,
                um_per_px=um_per_px_global,  # will be None if we never estimated µm/px
            )
            try:
                viewer.status = "✅ Loaded existing analysis."
            except Exception:
                pass
        except Exception as e:
            if parent is not None:
                QMessageBox.critical(
                    parent,
                    "Load data",
                    f"Failed to visualize analysis from:\n{out_dir}\n\nError:\n{e}",
                )
            try:
                viewer.status = f"❌ Failed to load analysis: {e}"
            except Exception:
                pass
  
    # ---------- RUN SEGMENTATION (pilot) ----------
    @run_seg_btn.changed.connect
    def _run_segmentation(_=None):
        viewer = current_viewer()
        if viewer is None:
            return
        if not form.output_folder.value:
            viewer.status = "⚠️ Please choose an Output Folder."
            return
        out_dir = Path(form.output_folder.value)

        all_cases = _gather_all_cases()
        if not all_cases:
            viewer.status = "⚠️ No inputs configured."
            return

        # let user choose which conditions to segment now
        chosen = _choose_conditions_dialog(viewer.window._qt_window, [(p, t) for p, t in all_cases])
        
        # Ask for channels once (use the first selected case as the sampler)
        base_for_sampling = chosen[0][0] if chosen else None
        if not base_for_sampling or not _ensure_channel_selection(base_for_sampling, viewer.window._qt_window):
            viewer.status = "⚠️ Could not detect/select channels."
            return

        # Use the user-selected channels everywhere downstream
        chan_list = chosen_channel_names[:]
        seg_name  = chosen_seg_channel
        ch_map    = chosen_channel_map

        print(f"Using channels: {chan_list}, segmenting on: {seg_name}")
        print(f"Channel rename map: {ch_map}")
        
        if not chosen:
            viewer.status = "ℹ️ Segmentation cancelled."
            return

        done_index = _scan_done_tags(out_dir)
        chosen_to_run = [(p, tag) for (p, tag) in chosen if tag not in done_index]

        if not chosen_to_run:
            viewer.status = "ℹ️ All selected cases already have outputs on disk. Nothing to do."
            return

        #n_workers = int(form.num_workers.value)
        n_workers = DEFAULT_WORKERS

        @thread_worker
        def _worker():
            total = len(chosen_to_run)
            bridge.total_changed.emit(0, total)
            for i, (p, tag) in enumerate(chosen_to_run, start=1):
                bridge.now_processing.emit(str(p), str(tag))
                bridge.file_changed.emit(0, 0)  # reset per-file bar (unknown granularity for pilot)
                current_viewer().status = f"Working: {tag} — {Path(p).name}"
                bridge.log_line.emit(f"Segmentation → {tag}: {p}")

                run_analysis(
                    str(p),
                    str(out_dir),
                    chan_list,
                    n_workers,
                    progress_callback=lambda d, t: bridge.file_changed.emit(d, t),
                    tag=tag,
                    save_extracted=False, #bool(form.save_extracted.value),
                    seg_channel=chosen_seg_channel,
                    channel_rename_map=chosen_channel_map,
                    seg_model=str(seg_model.value),
                    seg_diameter=int(seg_diameter.value),
                    seg_diam_units=str(seg_diam_units.value),
                    seg_scale=float(seg_scale.value),
                )

                bridge.total_changed.emit(i, total)
            bridge.finished.emit()
            return str(out_dir), chosen_to_run

        def _done(res):
            out_path, selected_cases = res
            viewer.status = "✅ Segmentation done. Adjust filters, then Run Analysis."
            # hook: capture user filter settings while they move sliders
            def _on_filter_change(ranges: Dict[str, tuple]):
                filter_ranges.clear()
                filter_ranges.update(ranges)
            try:
                show_analysis_results(
                    viewer,
                    out_path,
                    chan_list,
                    selected_cases,
                    initial_ranges=filter_ranges.copy(),   # seed if we had prior
                    on_filter_change=_on_filter_change,
                    show_filter=True,
                    show_boxplot=False,
                    show_radial_viewer=False,
                    show_pcc=False,
                    um_per_px=um_per_px_global,
                )
            except Exception as e:
                viewer.status = f"⚠️ Visualization warning: {e}"

        def _err(e):
            viewer.status = f"❌ Error: {e}"

        w = _worker(); w.returned.connect(_done); w.errored.connect(_err); w.start()

    # ---------- RUN ANALYSIS (full batch) ----------
    @run_all_btn.changed.connect
    def _run_all(_=None):
        viewer = current_viewer()
        if viewer is None:
            return
        if not form.output_folder.value:
            viewer.status = "⚠️ Please choose an Output Folder."
            return
        out_dir = Path(form.output_folder.value)

        all_cases = _gather_all_cases()
        if not all_cases:
            viewer.status = "⚠️ No inputs configured."
            return

        # IMPORTANT: do NOT filter by tag here. We want every case considered.
        to_run = list(all_cases)  # ← include pilot cases too

        # Ensure channels
        base_for_sampling = to_run[0][0] if to_run else form.input_folder.value
        if base_for_sampling and not _ensure_channel_selection(base_for_sampling, viewer.window._qt_window):
            viewer.status = "⚠️ Could not detect/select channels."
            return

        chan_list = chosen_channel_names[:]
        seg_name  = chosen_seg_channel
        ch_map    = chosen_channel_map

        if not to_run:
            viewer.status = "ℹ️ Everything already processed. Nothing to run."
            return

        #n_workers = int(form.num_workers.value)
        n_workers = DEFAULT_WORKERS

        # carry the thresholds captured during pilot
        fea_thresh = filter_ranges.copy()

        @thread_worker
        def _worker():
            total = len(to_run)
            bridge.total_changed.emit(0, total)
            for i, (p, tag) in enumerate(to_run, start=1):
                bridge.now_processing.emit(str(p), str(tag))
                bridge.file_changed.emit(0, 0)
                current_viewer().status = f"Working: {tag} — {Path(p).name}"
                bridge.log_line.emit(f"Analysis → {tag}: {p}")

                run_analysis(
                    str(p),
                    str(out_dir),
                    chan_list,
                    n_workers,
                    progress_callback=lambda d, t: bridge.file_changed.emit(d, t),
                    tag=tag,
                    save_extracted=False, #bool(form.save_extracted.value),
                    seg_channel=seg_name,
                    channel_rename_map=ch_map,
                    feature_thresholds=fea_thresh,
                    seg_model=str(seg_model.value),
                    seg_diameter=int(seg_diameter.value),
                    seg_diam_units=str(seg_diam_units.value),
                    seg_scale=float(seg_scale.value),
                )

                bridge.total_changed.emit(i, total)
            bridge.finished.emit()
            return str(out_dir), to_run

        def _done(res):
            out_path, tasks_with_tags = res
            viewer.status = "✅ Analysis completed."
            try:
                show_analysis_results(
                    viewer,
                    out_path,
                    chan_list,
                    tasks_with_tags,
                    initial_ranges=fea_thresh,
                    on_filter_change=None,
                    show_filter=False,
                    show_boxplot=True,
                    show_radial_viewer=True,
                    show_pcc=True,
                    um_per_px=um_per_px_global,
                )
            except Exception as e:
                viewer.status = f"⚠️ Visualization warning: {e}"

        def _err(e):
            viewer.status = f"❌ Error: {e}"

        w = _worker(); w.returned.connect(_done); w.errored.connect(_err); w.start()

    logo_path = Path(__file__).parent.parent.parent / "logo_isMap.png" 

    host = wrap_form_with_logo(ui.native, logo_path)
    return host


__all__ = ["tcell_widget"]
