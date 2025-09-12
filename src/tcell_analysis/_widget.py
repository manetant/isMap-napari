# src/tcell_analysis/_widget.py
from pathlib import Path
from dataclasses import dataclass

from magicgui import magicgui
from magicgui.widgets import Container, PushButton, FileEdit, Label

from napari.qt.threading import thread_worker
from napari.utils import progress
from napari import current_viewer

from qtpy.QtWidgets import QInputDialog, QApplication

from .analysis import run_analysis
from .visualization.view_in_napari import show_analysis_results
from typing import Dict, List, Optional, Tuple

import logging
logger = logging.getLogger(__name__)


@dataclass
class RowState:
    tag: str = ""          # the tag provided for this input
    tagged_path: str = ""  # which path that tag corresponds to (debounce)


def tcell_widget():
    # --- state ---
    primary_state = RowState()                    # track tag for primary input
    extra_rows: list[FileEdit] = []              # list of extra FileEdits
    state_by_row: dict[FileEdit, RowState] = {}  # tag/state for each extra

    # --- base form (your original fields, without call_button) ---
    @magicgui(
        input_folder={"widget_type": "FileEdit", "mode": "d", "label": "Input Folder"},
        output_folder={"widget_type": "FileEdit", "mode": "d", "label": "Output Folder"},
        channel_names={"widget_type": "LineEdit", "label": "Channels (comma-separated)", "value": "ICAM1,pTyr,Actin"},
        num_workers={"widget_type": "SpinBox", "min": 1, "max": 32, "value": 4},
        save_extracted={"widget_type": "CheckBox", "label": "Save extracted cell crops", "value": True},
        call_button=False,
    )
    def form(
        input_folder: Path,
        output_folder: Path,
        channel_names: str,
        num_workers: int,
        save_extracted: bool,
    ):
        pass

    # Prompt for tag once when the PRIMARY input is chosen (debounced by path)
    @form.input_folder.changed.connect
    def _on_primary_input_changed(path: Path | None):
        if not path:
            return
        cur = str(path)
        # Default output next to the first chosen input
        if not form.output_folder.value:
            form.output_folder.value = Path(cur).parent / "tcell-results"
        # Only prompt if this path wasn't tagged before
        if primary_state.tagged_path != cur:
            t, ok = QInputDialog.getText(
                form.native, "Tag this input", "Enter a tag (e.g., Treatment A):"
            )
            if ok and str(t).strip():
                primary_state.tag = str(t).strip()
                primary_state.tagged_path = cur

    # --- extra inputs UI ---
    add_btn = PushButton(label="+ Add input")

    # Start NON-scrollable to avoid first-click no-op; enable after first row is added
    extra_inputs_box = Container(layout="vertical", labels=True, scrollable=False)
    extra_inputs_box.native.setMaximumHeight(200)
    _ = extra_inputs_box.native  # realize early

    # --- run button (create before building ui) ---
    run_btn = PushButton(label="Run Analysis")

    # --- layout: Input → +Add → extras → Channels → Num workers → Output → Run ---
    ui = Container(layout="vertical", labels=True)
    ui.append(form.input_folder)   # Input Folder
    ui.append(add_btn)             # + Add input
    ui.append(extra_inputs_box)    # Extra Input Folders
    ui.append(form.channel_names)  # Channels
    ui.append(form.num_workers)    # Num workers
    ui.append(form.save_extracted) # Save extracted crops
    ui.append(form.output_folder)  # Output Folder
    ui.append(run_btn)             # Run Analysis
    _ = ui.native  # realize main container early

    # --- helper to add a new extra row ---
    def _make_extra_row():
        fe = FileEdit(mode="d", label="Choose Directory", value=None)
        _ = fe.native  # realize before insertion
        fe.native.setMinimumWidth(400)

        st = RowState()
        state_by_row[fe] = st

        @fe.changed.connect
        def _on_extra_changed(_=None):
            cur = str(fe.value or "")
            if not cur:
                return
            if not form.output_folder.value:
                form.output_folder.value = Path(cur).parent / "tcell-results"
            if st.tagged_path != cur:
                t, ok = QInputDialog.getText(
                    fe.native, "Tag this input", "Enter a tag (e.g., Treatment B):"
                )
                if ok and str(t).strip():
                    st.tag = str(t).strip()
                    st.tagged_path = cur

        # Create a label for the extra input
        label = Label(value=f"Extra Input {len(extra_rows) + 1}")

        # Put the FileEdit inside a small row container for more stable layouting
        row = Container(layout="horizontal", labels=False)
        row.append(label)
        row.append(fe)
        _ = row.native  # realize row container

        extra_rows.append(fe)
        extra_inputs_box.append(row)

        # Enable scrolling after the FIRST row appears
        if len(extra_rows) == 1 and not extra_inputs_box.scrollable:
            extra_inputs_box.scrollable = True
            # Realize the new scroll area widgets promptly
            _ = extra_inputs_box.native

        # --- force layout + paint so the row shows immediately on first click ---
        # Inner content (the extras box)
        if extra_inputs_box.native.layout() is not None:
            extra_inputs_box.native.layout().activate()
        extra_inputs_box.native.updateGeometry()
        extra_inputs_box.refresh()

        # Outer container
        if ui.native.layout() is not None:
            ui.native.layout().activate()
        ui.native.updateGeometry()
        ui.refresh()

        QApplication.processEvents()

    # Use MagicGUI's signal (not .native.clicked)
    @add_btn.changed.connect
    def _on_add_clicked():
        _make_extra_row()

    # --- run button ---
    @run_btn.changed.connect
    def _run_all():
        viewer = current_viewer()
        if viewer is None:
            return
        out_dir = form.output_folder.value
        if not out_dir:
            viewer.status = "⚠️ Please choose an Output Folder."
            return
        out_dir = Path(out_dir)
        chan_list = [c.strip() for c in form.channel_names.value.split(",") if c.strip()]
        num_workers = int(form.num_workers.value)

        tasks: list[tuple[Path, str]] = []

        # Primary input: require tag to already be set (no reprompt here)
        if not form.input_folder.value:
            viewer.status = "⚠️ Please choose the primary Input Folder."
            return
        if not primary_state.tag.strip():
            viewer.status = "⚠️ Tag missing for primary input. Click the Input Folder field again to set it."
            return
        tasks.append((Path(form.input_folder.value), primary_state.tag))

        # Extra inputs: require tag to already be set (no reprompt here)
        for fe in extra_rows:
            if not fe.value:
                viewer.status = "⚠️ One of the extra inputs is empty."
                return
            st = state_by_row.get(fe)
            if not st or not st.tag.strip():
                viewer.status = "⚠️ Tag missing for one of the extra inputs. Click that Input Folder to set it."
                return
            tasks.append((Path(fe.value), st.tag))

        viewer.status = "⚙️ Running analysis for: " + ", ".join(p.name for p, _ in tasks)

        @thread_worker
        def _worker():
            pbar = progress(desc="T Cell Analysis (batch)", total=len(tasks))
            for i, (in_dir, tag) in enumerate(tasks, start=1):
                run_analysis(
                    str(in_dir),
                    str(out_dir),  # single shared output folder
                    chan_list,
                    num_workers,
                    progress_callback=None,
                    tag=tag,
                    save_extracted=bool(form.save_extracted.value),
                )
                pbar.n = i
                pbar.refresh()
            return str(out_dir), tasks

        def _done(result):
            out_path, tasks_with_tags = result
            viewer.status = "✅ Batch completed."
            try:
                show_analysis_results(viewer, out_path, chan_list, tasks_with_tags)
            except Exception as e:
                viewer.status = f"⚠️ Visualization warning: {e}"

        def _err(e):
            viewer.status = f"❌ Error: {e}"

        w = _worker()
        w.returned.connect(_done)
        w.errored.connect(_err)
        w.start()

    return ui


__all__ = ["tcell_widget"]
