# src/tcell_analysis/_widget.py
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import os
import pandas as pd

from magicgui import magicgui
from magicgui.widgets import Container, PushButton, FileEdit, Label, ComboBox, SpinBox

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



def _find_sample_image(base: Path) -> Path | None:
    # search inside base for a single image we can open to detect channels
    exts = (".vsi", ".nd2", ".czi", ".lif", ".tif", ".tiff")
    for ext in exts:
        hits = list(base.rglob(f"*{ext}"))
        if hits:
            return hits[0]
    return None

def _choose_channels_and_seg(parent, file_path: Path, prefer_name: str = "Actin"):
    """
    Pop a table listing detected channels:
      (selected_names_ordered, channel_map_orig_to_new, segmentation_name) 
    or (None, None, None) if cancelled.
    """
    # detect channels with BioImage
    img = BioImage(str(file_path))
    detected = list(getattr(img, "channel_names", []) or [])
    if not detected:
        # fallback to OME-XML or generic C0.. if needed
        with tiff.TiffFile(str(file_path)) as tf:
            chs = tf.pages[0].tags.get("ImageDescription", None)
        # just make something reasonable if none
        detected = [f"C{i}" for i in range(int(img.shape["C"]))]

    dlg = QDialog(parent)
    dlg.setWindowTitle("Select & Rename Channels, Pick Segmentation Channel")
    lay = QVBoxLayout(dlg)

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
        return None, None, None

    # collect
    keep_rows = []
    for r in range(table.rowCount()):
        keep = table.cellWidget(r, 0).isChecked()
        orig = table.item(r, 1).text()
        new  = table.cellWidget(r, 2).text().strip() or orig
        if keep:
            keep_rows.append((orig, new, r))

    if not keep_rows:
        return None, None, None

    seg_row = seg_group.checkedId()
    seg_orig = detected[seg_row] if 0 <= seg_row < len(detected) else keep_rows[-1][0]
    # If segmentation channel wasn’t kept, force it to the first kept row
    if not any(orig == seg_orig for (orig, _, _) in keep_rows):
        seg_orig = keep_rows[0][0]

    selected_names = [new for (_, new, _) in keep_rows]
    channel_map = {orig: new for (orig, new, _) in keep_rows}

    # final segmentation name is the renamed one (if kept)
    seg_new = channel_map.get(seg_orig, keep_rows[0][1])
    return selected_names, channel_map, seg_new


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
    txt, ok = QInputDialog.getText(parent.native if hasattr(parent, "native") else parent, title, label, text=default)
    if ok and str(txt).strip():
        return str(txt).strip()
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
        self.lblOverall = QLabel("Overall: 0 / 0")
        self.pbOverall = QProgressBar()
        self.pbOverall.setMinimum(0); self.pbOverall.setMaximum(100)
        self.lblCurrent = QLabel("Current: –")
        self.pbCurrent = QProgressBar()
        self.pbCurrent.setMinimum(0); self.pbCurrent.setMaximum(100)
        self.lblTag = QLabel("Tag: –")
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(160)
        lay.addWidget(self.lblOverall, 0, 0, 1, 2)
        lay.addWidget(self.pbOverall, 1, 0, 1, 2)
        lay.addWidget(self.lblCurrent, 2, 0, 1, 2)
        lay.addWidget(self.pbCurrent, 3, 0, 1, 2)
        lay.addWidget(self.lblTag, 4, 0, 1, 2)
        lay.addWidget(self.log, 5, 0, 1, 2)

    def set_total(self, done, total):
        pct = 0 if total == 0 else int(100 * done / total)
        self.pbOverall.setValue(pct)
        self.lblOverall.setText(f"Overall: {done} / {total}")

    def set_file(self, done, total):
        pct = 0 if total == 0 else int(100 * done / total)
        self.pbCurrent.setValue(pct)

    def set_now(self, path, tag):
        self.lblCurrent.setText(f"Current: {Path(path).name}")
        self.lblTag.setText(f"Tag: {tag}")

    def append(self, msg):
        self.log.append(msg)
        self.log.ensureCursorVisible()


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

    def _ensure_channel_selection(base_path_for_sampling: Path, parent_window) -> bool:
        nonlocal chosen_channel_names, chosen_seg_channel, chosen_channel_map

        if chosen_channel_names and chosen_seg_channel:
            return True

        sample = _find_sample_image(base_path_for_sampling)
        if sample is None:
            return False

        sel_names, ch_map, seg_new = _choose_channels_and_seg(parent_window, sample, prefer_name="Actin")
        if not sel_names:
            return False

        chosen_channel_names = sel_names
        chosen_seg_channel   = seg_new
        chosen_channel_map   = ch_map
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
        seg_scale: float,        
    ):
        ...

    # ---- Segmentation Parameters Section ----
    seg_header = Label(value="<b>Segmentation Parameters</b> "
                             "<span style='color:#777'>(used only for segmentation)</span>")
    seg_model = ComboBox(label="Model", choices=["cyto3", "cyto2"], value="cyto3")
    seg_diameter = SpinBox(label="Diameter (px)", min=5, max=2000, value=100, step=1)
    seg_scale = ComboBox(
        label="Scale (×)",
        choices=[f"{x/10:.1f}" for x in range(10, 2, -1)],
        value="1.0",
        tooltip="Scale applied to segmentation (1.0 = native size)",
    )

    seg_block = Container(layout="vertical", labels=True)
    seg_block.extend([seg_header, seg_model, seg_diameter, seg_scale])

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
    '''
    # ---------- EXTRA INPUTS ----------
    add_btn = PushButton(label="+ Add input")
    extra_box = Container(layout="vertical", labels=True, scrollable=False)
    extra_box.native.setMaximumHeight(220)

    def _make_extra_row():
        fe = FileEdit(mode="d", label="Input Folder", value=None)
        fe.native.setMinimumWidth(420)
        st = RowState()
        state_by_row[fe] = st

        @fe.changed.connect
        def _on_extra(_=None):
            cur = str(fe.value or "")
            if not cur:
                return
            if not form.output_folder.value:
                form.output_folder.value = Path(cur) / "tcell-results"
            subs = _iter_immediate_subfolders(Path(cur))
            if subs:
                st.conditions_by_subfolder = _ask_conditions_for_subfolders(fe, Path(cur), st.conditions_by_subfolder if st.conditions_for_path == cur else None)
                st.conditions_for_path = cur
                st.tag = ""; st.tagged_path = ""
            else:
                if st.tagged_path != cur:
                    default = Path(cur).name
                    t = _prompt_text(fe, "Tag this input", f"Enter a tag for “{default}”:", default)
                    st.tag = t or ""
                    st.tagged_path = cur
                st.conditions_by_subfolder.clear(); st.conditions_for_path = ""

        row = Container(layout="horizontal", labels=False)
        row.append(Label(value=f"Extra Input {len(extra_rows)+1}"))
        row.append(fe)
        extra_rows.append(fe)
        extra_box.append(row)

        if len(extra_rows) == 1 and not extra_box.scrollable:
            extra_box.scrollable = True

        for w in (extra_box.native,):
            if w.layout() is not None:
                w.layout().activate()
            w.updateGeometry()
        QApplication.processEvents()

    @add_btn.changed.connect
    def _on_add(_=None):
        _make_extra_row()
    '''

    # ---------- BUILD UI ORDER ----------
    run_seg_btn = PushButton(label="Run Segmentation")
    run_all_btn = PushButton(label="Run Analysis")

    ui = Container(layout="vertical", labels=True)
    ui.append(form.input_folder)
    #ui.append(add_btn)
    #ui.append(extra_box)
    #ui.append(form.num_workers)
    #ui.append(form.save_extracted)
    ui.append(form.output_folder)
    ui.append(seg_block)
    ui.append(run_seg_btn)   # first stage
    ui.append(run_all_btn)   # second stage


    # ---- Progress dock (one per widget) ----
    progress_panel = ProgressPanel()
    bridge = _ProgressBridge()

    # connect signals to the panel (main thread safe)
    bridge.total_changed.connect(progress_panel.set_total)
    bridge.file_changed.connect(progress_panel.set_file)
    bridge.now_processing.connect(progress_panel.set_now)
    bridge.log_line.connect(progress_panel.append)

    # mount the dock (left or right as you prefer)
    try:
        current_viewer().window.add_dock_widget(progress_panel, area="left", name="T-Cell Analysis Progress")
    except Exception:
        pass

    # ---------- GATHER ALL CASES (utility) ----------
    '''
    def _gather_all_cases() -> List[Tuple[Path, str]]:
        tasks: List[Tuple[Path, str]] = []

        # primary
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

        # extras
        for fe in extra_rows:
            if not fe.value:
                continue
            st = state_by_row.get(fe, RowState())
            base = Path(fe.value)
            subs = _iter_immediate_subfolders(base)
            if subs:
                if not st.conditions_by_subfolder:
                    st.conditions_by_subfolder = _ask_conditions_for_subfolders(fe, base, None)
                    st.conditions_for_path = str(base)
                for s in sorted(subs):
                    tasks.append((s, st.conditions_by_subfolder.get(s.name, s.name)))
            else:
                if not st.tag.strip():
                    t = _prompt_text(fe, "Tag this input", f"Enter a tag for “{base.name}”:", base.name)
                    if t:
                        st.tag = t; st.tagged_path = str(base)
                        state_by_row[fe] = st
                if st.tag.strip():
                    tasks.append((base, st.tag))
        return tasks
    '''

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
                )
            except Exception as e:
                viewer.status = f"⚠️ Visualization warning: {e}"

        def _err(e):
            viewer.status = f"❌ Error: {e}"

        w = _worker(); w.returned.connect(_done); w.errored.connect(_err); w.start()

    return ui


__all__ = ["tcell_widget"]
