import glob
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from magicgui import magicgui
from magicgui.widgets import Container, FileEdit, PushButton
from tifffile import imread

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from qtpy.QtWidgets import QWidget, QVBoxLayout


def show_analysis_results(
    viewer,
    output_folder,
    channel_names,
    tasks_with_tags,
    text_feature=None,
    text_size=11,
    text_color="yellow",
    rgb=False,
):
    # safe default
    if text_feature is None:
        text_feature = ["circularity"]

    # --- Build robust mapping from inputs to tags ---
    inputs_info = []
    for in_path, tag in tasks_with_tags:
        p = Path(in_path)
        try:
            p_res = p.resolve()
        except Exception:
            p_res = p
        inputs_info.append(
            {
                "path": p_res,
                "name": p.name,
                "stem": p.stem,
                "tag": tag,
            }
        )

    print(f"[INFO] Loading results from: {output_folder}")
    print(tasks_with_tags)

    root = Path(output_folder)

    # Find all case directories that actually contain results
    csv_hits = sorted(root.rglob("per_cell_features.csv"))
    mask_hits = sorted(root.rglob("Actin_mask.tiff"))

    if not csv_hits and not mask_hits:
        raise ValueError(
            f"No results found under {root}.\n"
            "Looked for 'per_cell_features.csv' and 'Actin_mask.tiff' recursively."
        )

    # Unique parent directories that hold each frame
    frame_dirs = sorted({p.parent for p in (csv_hits + mask_hits)})

    # --- Infer a tag per frame_dir with multiple heuristics ---
    def infer_tag_for_frame_dir(fd: Path) -> str:
        """Return the most plausible tag for a given frame directory."""
        try:
            fd_res = fd.resolve()
        except Exception:
            fd_res = fd

        # 1) Exact ancestor name match (basename or stem of input)
        ancestor_names = {anc.name for anc in [fd_res, *fd_res.parents]}
        anc_lower = {n.casefold() for n in ancestor_names}
        for info in inputs_info:
            if info["name"].casefold() in anc_lower or info["stem"].casefold() in anc_lower:
                return info["tag"]

        # 2) Input path containment (the output lives under the input folder tree)
        for info in inputs_info:
            try:
                common = Path(os.path.commonpath([str(fd_res), str(info["path"])]))
                if common == info["path"]:
                    return info["tag"]
            except Exception:
                pass

        # 3) Any path part contains the input name/stem (case-insensitive substring)
        parts_lower = [part.casefold() for part in fd_res.parts]
        for info in inputs_info:
            name_l, stem_l = info["name"].casefold(), info["stem"].casefold()
            if any((name_l in part) or (stem_l in part) for part in parts_lower):
                return info["tag"]

        # 4) Tag itself appears in the path (some pipelines write tags into output dirs)
        for info in inputs_info:
            tag_l = str(info["tag"]).casefold()
            if any(tag_l in part for part in parts_lower):
                return info["tag"]

        # No match
        print(
            f"[TAG] N/A for frame_dir={fd_res}; inputs checked="
            f"{[(i['path'], i['name'], i['stem'], i['tag']) for i in inputs_info]}"
        )
        return "N/A"

    frame_names = [fd.name for fd in frame_dirs]
    frame_tags = [infer_tag_for_frame_dir(fd) for fd in frame_dirs]

    image_stack = []
    mask_stack = []

    all_coords = []
    all_texts = []
    all_properties_lists = {}
    all_columns = set()

    def _normalize_to_uint8(img: np.ndarray) -> np.ndarray:
        """Normalize image to uint8 regardless of channel order (C,H,W) or (H,W,C) or 2D."""
        if img.ndim == 2:
            norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            return norm
        if img.ndim == 3:
            # heuristic: channel axis is whichever has size <= 8 typically
            # fallback to assuming channels-first if ambiguous
            c_axis = 0 if img.shape[0] <= 8 else (2 if img.shape[-1] <= 8 else 0)
            # move channels to axis 0
            if c_axis == 2:
                img_c0 = np.transpose(img, (2, 0, 1))
            elif c_axis == 0:
                img_c0 = img
            else:
                img_c0 = img  # safe default

            out = np.zeros_like(img_c0, dtype=np.uint8)
            for c in range(img_c0.shape[0]):
                out[c] = cv2.normalize(img_c0[c], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # return in the same layout as input
            if c_axis == 2:
                return np.transpose(out, (1, 2, 0))
            else:
                return out
        raise ValueError(f"Unsupported image ndim={img.ndim}")

    def load_frame_data(frame_dir):
        frame_dir = Path(frame_dir)
        frame_name = frame_dir.name

        # The main multi-channel image: try "<frame_name>.tiff" first; otherwise any *.tif*
        image_path = frame_dir / f"{frame_name}.tiff"
        if not image_path.exists():
            candidates = [
                p for p in frame_dir.glob("*.tif*")
                if p.name.lower() != "actin_mask.tiff" and "mask" not in p.name.lower()
            ]
            if not candidates:
                raise FileNotFoundError(f"No multi-channel TIFF found in {frame_dir}")
            image_path = sorted(candidates)[0]

        mask_path = frame_dir / "Actin_mask.tiff"
        csv_path = frame_dir / "per_cell_features.csv"

        for path in [image_path, mask_path, csv_path]:
            if not path.exists():
                raise FileNotFoundError(f"Missing required file: {path}")

        image = imread(str(image_path))
        image_uint8 = _normalize_to_uint8(image)

        mask = imread(str(mask_path))
        df = pd.read_csv(csv_path).set_index("label")
        return image_uint8, mask, df

    # Discover union of all properties
    for frame_dir in frame_dirs:
        _, _, df = load_frame_data(frame_dir)
        all_columns.update(df.columns)

    all_columns = sorted(all_columns)
    for col in all_columns:
        all_properties_lists[col] = []

    # Load frames and build coordinates/properties/text
    for frame_index, frame_dir in enumerate(frame_dirs):
        image, mask, df = load_frame_data(frame_dir)

        # Ensure images become (H,W,C) for rgb=True path or per-channel display
        if image.ndim == 3 and image.shape[0] < image.shape[-1]:
            # channels-first → to channels-last
            image = np.transpose(image, (1, 2, 0))
        elif image.ndim == 2:
            image = image[..., np.newaxis]

        image_stack.append(image)
        mask_stack.append(mask)

        for label_id in df.index:
            row = df.loc[label_id]
            y, x = float(row["y"]), float(row["x"])
            all_coords.append((frame_index, y, x))
            for col in all_columns:
                val = row.get(col, np.nan)
                all_properties_lists[col].append(val)

            lines = [f"ID {label_id}"]
            for feature in text_feature:
                val = row.get(feature, "N/A")
                if pd.isna(val):
                    val_str = "N/A"
                elif isinstance(val, (float, int)):
                    val_str = f"{val:.2f}"
                else:
                    val_str = str(val)
                lines.append(f"{feature}: {val_str}")
            all_texts.append("\n".join(lines))

    if len(image_stack) == 0:
        raise ValueError("No images found to display. Did analysis finish successfully?")
    else:
        image_stack = np.stack(image_stack, axis=0)

    if len(mask_stack) > 0:
        mask_stack = np.stack(mask_stack, axis=0)
    else:
        print("[WARN] No masks found to display.")
        mask_stack = None

    all_coords_np = np.array(all_coords)
    all_texts_list = list(all_texts)
    # ✅ fixed dict-comprehension
    all_properties = {col: np.array(vals) for col, vals in all_properties_lists.items()}

    # Keep originals for filtering
    original_coords_np = all_coords_np.copy()
    original_texts_list = all_texts_list.copy()
    original_properties = {k: v.copy() for k, v in all_properties.items()}

    # --- Add image layers ---
    if rgb:
        if image_stack.shape[-1] < 3:
            for i, name in enumerate(channel_names):
                viewer.add_image(
                    image_stack[..., i],
                    name=name,
                    colormap="gray",
                    blending="additive",
                    opacity=1.0,
                )
        else:
            viewer.add_image(
                image_stack[..., :3],
                name="RGB Composite",
                rgb=True,
                opacity=1.0,
            )
    else:
        for i, name in enumerate(channel_names):
            viewer.add_image(
                image_stack[..., i],
                name=name,
                colormap="gray",
                blending="additive",
                opacity=1.0,
            )

    # --- Add labels ONCE ---
    if mask_stack is not None:
        viewer.add_labels(mask_stack, name="Mask Stack")

    # --- Points layer with properties/text ---
    points = viewer.add_points(
        original_coords_np,
        name="Cell Labels",
        size=5,
        face_color="transparent",
        properties=original_properties,
    )
    points.edge_color = "red"
    points.text = {
        "string": original_texts_list,
        "size": text_size,
        "color": text_color,
        "anchor": "center",
    }

    # --- Overlay that updates when the user changes frame ---
    def _update_overlay(event=None):
        # the leading axis is the "frame" axis we stacked on
        if viewer.dims.ndim == 0:
            idx = 0
        else:
            idx = int(viewer.dims.current_step[0]) if len(viewer.dims.current_step) > 0 else 0

        idx = max(0, min(idx, len(frame_names) - 1))
        viewer.text_overlay.text = f"Frame: {frame_names[idx]}   |   Group: {frame_tags[idx]}"
        viewer.text_overlay.position = "top_left"
        viewer.text_overlay.visible = True

    viewer.dims.events.current_step.connect(_update_overlay)
    _update_overlay()

    # --- Filtering widget ---
    @magicgui(
        auto_call=True,
        circularity={
            "label": "Circularity",
            "widget_type": "FloatSlider",
            "min": 0,
            "max": 1,
            "step": 0.1,
            "value": 0.3,
        },
        eccentricity={
            "label": "Eccentricity",
            "widget_type": "FloatSlider",
            "min": 0,
            "max": 1,
            "step": 0.1,
            "value": 0,
        },
        diameter={
            "label": "Diameter",
            "widget_type": "FloatSlider",
            "min": 0,
            "max": 200,
            "step": 10,
            "value": 20,
        },
    )
    def filter_points(circularity, eccentricity, diameter):
        if (
            "circularity" not in original_properties
            or "eccentricity" not in original_properties
            or "equivalent_diameter" not in original_properties
        ):
            print("[WARNING] Required property missing. Cannot filter.")
            return

        circ = original_properties["circularity"]
        ecc = original_properties["eccentricity"]
        diam = original_properties["equivalent_diameter"]

        mask = (circ >= circularity) & (ecc >= eccentricity) & (diam >= diameter)

        if mask.shape[0] != original_coords_np.shape[0]:
            print("[WARNING] Property mask length mismatch with original data. Skipping filter.")
            return

        points.data = original_coords_np[mask]
        points.text.values = [original_texts_list[i] for i in range(len(original_texts_list)) if mask[i]]
        points.properties = {k: v[mask] for k, v in original_properties.items()}

    viewer.window.add_dock_widget(filter_points, area="right")

    # Figure + canvas embedded in Qt
    fig, ax = plt.subplots()
    canvas = FigureCanvas(fig)
    panel = QWidget()
    _panel_layout = QVBoxLayout(panel)
    _panel_layout.setContentsMargins(6, 6, 6, 6)
    _panel_layout.addWidget(canvas)

    # Find available mean-intensity metrics from properties (e.g., "Actin_mean_intensity")
    mean_intensity_metrics = [k for k in original_properties.keys() if k.endswith("_mean_intensity")]
    default_metric = "Actin_mean_intensity" if "Actin_mean_intensity" in mean_intensity_metrics else (mean_intensity_metrics[0] if mean_intensity_metrics else None)

    def _plot_box(metric: str | None):
        ax.clear()
        if metric is None or metric not in points.properties:
            ax.set_title("No mean-intensity metric found")
            canvas.draw_idle()
            return

        # CURRENT filtered points only
        coords = points.data
        if coords.size == 0:
            ax.set_title("No points after filtering")
            canvas.draw_idle()
            return

        frame_idx = coords[:, 0].astype(int)
        intens = np.asarray(points.properties[metric])
        # Map frame index -> tag
        Groups = [frame_tags[i] for i in frame_idx]

        df_plot = pd.DataFrame({"Group": Groups, metric: intens})
        # Ensure deterministic order by tag
        groups = []
        labels = []
        for tag, g in df_plot.groupby("Group", sort=True):
            groups.append(g[metric].values)
            labels.append(str(tag))

        if len(groups) == 0:
            ax.set_title("No data to plot")
            canvas.draw_idle()
            return

        ax.boxplot(groups, labels=labels, showfliers=True)
        ax.set_xlabel("Group")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by Group")
        for label in ax.get_xticklabels():
            label.set_rotation(20)
            label.set_ha("right")
        fig.tight_layout()
        canvas.draw_idle()

    # Control widget to pick metric + auto-refresh
    @magicgui(
        auto_call=True,
        metric={"label": "Metric", "choices": mean_intensity_metrics, "value": default_metric},
    )
    def boxplot_controls(metric: str = default_metric):
        _plot_box(metric)

    # Keep the plot in sync when filters change (points layer updates)
    def _on_points_changed(event=None):
        # Reuse current selection in the control
        cur_metric = boxplot_controls.metric.value if hasattr(boxplot_controls, "metric") else default_metric
        _plot_box(cur_metric)

    points.events.data.connect(_on_points_changed)
    # Initial draw
    _plot_box(default_metric)

    def _safe_add_dock(widget, name, area="right"):
        # Prefer public API
        docks = getattr(viewer.window, "dock_widgets", [])
        try:
            # NapariDockWidget usually has `.name`; Qt dock has `.windowTitle()`
            for dw in list(docks):
                dw_name = getattr(dw, "name", None)
                if dw_name is None:
                    try:
                        dw_name = dw.windowTitle()
                    except Exception:
                        dw_name = None
                if dw_name == name:
                    viewer.window.remove_dock_widget(dw)
        except Exception:
            # Fallback for very old napari versions (no .dock_widgets)
            pass

        viewer.window.add_dock_widget(widget, area=area, name=name)

    _safe_add_dock(panel, "Intensity Boxplot")
    _safe_add_dock(boxplot_controls, "Boxplot Controls")
    # Add both the plot and its controls to the napari UI
    #viewer.window.add_dock_widget(panel, area="right", name="Intensity Boxplot")
    #viewer.window.add_dock_widget(boxplot_controls, area="right", name="Boxplot Controls")

    # --- Export UI ---
    default_export_path = Path(output_folder) / "filtered_points_export.csv"
    save_path_picker = FileEdit(label="Save CSV", mode="w", value=str(default_export_path))
    export_button = PushButton(label="Export CSV")

    def _do_export():
        if points.data.shape[0] == 0:
            print("[INFO] No points to export.")
            return

        output_path = str(save_path_picker.value)
        if not output_path.endswith(".csv"):
            output_path += ".csv"

        filtered_coords = points.data
        filtered_properties = points.properties

        data = {
            "frame_index": filtered_coords[:, 0],
            "y": filtered_coords[:, 1],
            "x": filtered_coords[:, 2],
        }
        for prop_name, prop_vals in filtered_properties.items():
            data[prop_name] = prop_vals

        frame_names_from_idx = [frame_names[int(idx)] for idx in filtered_coords[:, 0]]
        data["frame_name"] = frame_names_from_idx
        data["frame_tag"] = [frame_tags[int(idx)] for idx in filtered_coords[:, 0]]

        df_out = pd.DataFrame(data)
        df_out.to_csv(output_path, index=False)
        viewer.status = f"✅ Exported filtered points to: {output_path}"

    export_button.changed.connect(_do_export)
    export_widget = Container(widgets=[save_path_picker, export_button])
    viewer.window.add_dock_widget(export_widget, area="right", name="Export Filtered Points")
