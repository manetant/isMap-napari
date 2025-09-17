# --- paste the whole function below, keep your imports at top ---
from pathlib import Path
import numpy as np
import pandas as pd
from tifffile import imread
import dask.array as da
from dask import delayed
from magicgui import magicgui
from magicgui.widgets import Container, FileEdit, PushButton
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from qtpy.QtWidgets import QWidget, QVBoxLayout

def show_analysis_results(
    viewer,
    output_folder,
    channel_names,
    tasks_with_tags,             # kept for API compatibility (unused now)
    text_feature=None,
    text_size=11,
    text_color="yellow",
    rgb=False,
    initial_ranges=None,           # NEW: dict like {"circularity": (0.3, 1.0), ...}
    on_filter_change=None, 
    show_filter: bool = True,
    show_boxplot: bool = True,
    export_csv: bool = True,
):
    if initial_ranges is None:
        initial_ranges = {}

    # ---------- helpers ----------
    def _remove_dock(name_or_widget):
        # napari supports removing by name string or widget
        try:
            viewer.window.remove_dock_widget(name_or_widget)
            return
        except Exception:
            pass
        # fallback: look into private registry if present
        try:
            d = getattr(viewer.window, "_dock_widgets", {})
            if isinstance(name_or_widget, str) and name_or_widget in d:
                viewer.window.remove_dock_widget(d[name_or_widget])
        except Exception:
            pass

    def compute_clims_per_channel(img):
        """img is (H,W) or (H,W,C) channels-last; returns list of (low, high)."""
        if img.ndim == 2:
            p1, p99 = np.percentile(img, (1, 99))
            return [(float(p1), float(p99))]
        C = img.shape[-1]
        clims = []
        for c in range(C):
            p1, p99 = np.percentile(img[..., c], (1, 99))
            clims.append((float(p1), float(p99)))
        return clims

    def find_image_path(frame_dir: Path) -> Path:
        """Prefer <frame_name>.tiff, otherwise first non-mask *.tif* in frame_dir."""
        name = frame_dir.name
        p = frame_dir / f"{name}.tiff"
        if p.exists():
            return p
        cands = [
            q for q in frame_dir.glob("*.tif*")
            if q.name.lower() != "actin_mask.tiff" and "mask" not in q.name.lower()
        ]
        if not cands:
            raise FileNotFoundError(f"No multi-channel TIFF in {frame_dir}")
        return sorted(cands)[0]

    def _eager_load_image_for_shape(path: Path):
        """Load image and return channels-last (H,W,C)."""
        arr = imread(str(path))
        if arr.ndim == 3 and arr.shape[0] < arr.shape[-1]:
            # channels-first -> channels-last
            arr = np.transpose(arr, (1, 2, 0))
        elif arr.ndim == 2:
            arr = arr[..., None]
        return arr

    def lazy_image_array(path: Path) -> da.Array:
        """Lazy dask array of (H,W,C) for napari, using one eager read for shape/dtype."""
        sample = _eager_load_image_for_shape(path)
        return da.from_delayed(
            delayed(_eager_load_image_for_shape)(path),
            shape=sample.shape,
            dtype=sample.dtype,
        )

    def load_mask_and_df(fd: Path):
        """Load mask + per_cell_features.csv and return (mask, df_with_index, tag)."""
        mask_path = fd / "Actin_mask.tiff"
        csv_path  = fd / "per_cell_features.csv"
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

        mask = imread(str(mask_path))
        df   = pd.read_csv(csv_path)

        # minimal schema guard
        for req in ("label", "y", "x"):
            if req not in df.columns:
                raise ValueError(f"{csv_path} missing required column: {req}")

        tag = str(df["tag"].iloc[0]) if "tag" in df.columns and len(df) else "N/A"
        df = df.set_index("label")
        return mask, df, tag

    # ---------- discover frames ----------
    root = Path(output_folder)
    mask_hits = sorted(root.rglob("Actin_mask.tiff"))
    if not mask_hits:
        raise ValueError(f"No Actin_mask.tiff under {root}")

    frame_dirs = sorted({p.parent for p in mask_hits})

    # Load lightweight things eagerly (mask/CSV/tag) and collect image paths
    image_paths, masks, dfs, frame_tags = [], [], [], []
    for fd in frame_dirs:
        try:
            mask, df, tag = load_mask_and_df(fd)
        except (FileNotFoundError, ValueError) as e:
            print(f"[WARN] Skipping {fd}: {e}")
            continue
        masks.append(mask)
        dfs.append(df)
        frame_tags.append(tag)
        try:
            image_paths.append(find_image_path(fd))
        except FileNotFoundError as e:
            print(f"[WARN] Skipping image for {fd}: {e}")
            image_paths.append(None)

    if len(dfs) == 0:
        raise ValueError("Found frames, but none had both mask and per_cell_features.csv.")

    # Keep list of frame names for overlay/exports
    frame_names = [Path(p).parent.name if p is not None else Path(dfs[i].index.name or f"frame_{i}").name
                   for i, p in enumerate(image_paths)]

    # ---------- build lazy image stack ----------
    # Some frames might be missing an image; filter those out consistently
    valid_idxs = [i for i, p in enumerate(image_paths) if p is not None]
    if not valid_idxs:
        raise ValueError("No frame images found to display.")

    image_arrays = [lazy_image_array(image_paths[i]) for i in valid_idxs]  # each (H,W,C)
    image_stack  = da.stack(image_arrays, axis=0)                          # (F,H,W,C) over valid frames

    # Use first valid frame for contrast limits
    first_frame = _eager_load_image_for_shape(image_paths[valid_idxs[0]])
    clims = compute_clims_per_channel(first_frame)

    # ---------- add image layers (one per channel) ----------
    num_channels = int(image_stack.shape[-1])
    # names for channels
    ch_names = list(channel_names)[:num_channels] + [f"Ch{c}" for c in range(len(channel_names), num_channels)]

    if rgb and num_channels >= 3:
        lows  = [clims[i][0] for i in range(min(3, len(clims)))]
        highs = [clims[i][1] for i in range(min(3, len(clims)))]
        viewer.add_image(
            image_stack[..., :3],
            name="RGB Composite",
            rgb=True,
            opacity=1.0,
            contrast_limits=(min(lows), max(highs)) if clims else None,
        )
    # always add separate grayscale layers
    for c in range(num_channels):
        viewer.add_image(
            image_stack[..., c],  # (F,H,W)
            name=ch_names[c],
            colormap="gray",
            blending="additive",
            opacity=1.0,
            contrast_limits=clims[c] if c < len(clims) else None,
            metadata={"channel_index": c},
        )

    # ---------- labels stack ----------
    # Only include masks for valid image frames (to keep leading axis aligned)
    masks_valid = [masks[i] for i in valid_idxs]
    mask_stack = np.stack(masks_valid, axis=0)
    if not np.issubdtype(mask_stack.dtype, np.integer):
        mask_stack = mask_stack.astype(np.int32, copy=False)
    viewer.add_labels(mask_stack, name="Mask Stack")

    # ---------- points/properties/text (vectorized) ----------
    # Align dfs and tags to valid_idxs too
    dfs_valid  = [dfs[i] for i in valid_idxs]
    tags_valid = [frame_tags[i] for i in valid_idxs]
    names_valid = [frame_dirs[i].name for i in valid_idxs]

    # union of columns across frames
    all_columns = sorted(set().union(*[set(df.columns) for df in dfs_valid]))
    prop_chunks = {col: [] for col in all_columns}
    all_coords = []
    all_texts = []

    if text_feature is None:
        text_feature = ["circularity"]
    feat_cols = text_feature if isinstance(text_feature, (list, tuple)) else [text_feature]
    fmt = "ID {label}\n" + "\n".join(f"{f}: {{{f}}}" for f in feat_cols if f)

    for fi, df in enumerate(dfs_valid):
        # coords: (N,3) = (frame, y, x)
        yx = df[["y", "x"]].to_numpy(dtype=float, copy=False)
        z  = np.full((len(df), 1), fi, dtype=float)
        coords = np.concatenate([z, yx], axis=1)
        all_coords.append(coords)

        # properties
        for col in all_columns:
            if col in df.columns:
                prop_chunks[col].append(df[col].to_numpy(copy=False))
            else:
                prop_chunks[col].append(np.full((len(df),), np.nan))

        # texts
        df_reset = df.reset_index()
        if fmt.endswith("ID {label}\n"):  # no feature lines requested
            texts = df_reset.apply(lambda r: f"ID {r['label']}", axis=1)
        else:
            texts = df_reset.apply(lambda r: fmt.format(**{k: str(v) for k, v in r.items()}), axis=1)
        all_texts.extend(list(texts))

    all_coords_np  = np.vstack(all_coords)
    all_properties = {col: np.concatenate(chunks) for col, chunks in prop_chunks.items()}
    texts_list     = list(all_texts)

    points = viewer.add_points(
        all_coords_np,
        name="Cell Labels",
        size=5,
        face_color="none",
        properties=all_properties,
    )
    points.edge_color = "red"
    points.text = {"string": texts_list, "size": text_size, "color": text_color, "anchor": "center"}

    # ---------- overlay using CSV tag ----------
    viewer.text_overlay.visible = True
    viewer.text_overlay.position = "top_left"

    def _update_overlay(event=None):
        if viewer.dims.ndim and viewer.dims.current_step:
            idx = int(viewer.dims.current_step[0])
        else:
            idx = 0
        idx = max(0, min(idx, len(names_valid) - 1))
        n_cells = int((mask_stack[idx] > 0).sum())
        viewer.text_overlay.text = f"Frame: {names_valid[idx]} | Group: {tags_valid[idx]} | Cells: {n_cells}"

    viewer.dims.events.current_step.connect(_update_overlay)
    _update_overlay()

    # ---------- filtering widget (range sliders are nicer, but keep your sliders) ----------
    # defaults if not provided
    if show_filter:
        c_rng = tuple(initial_ranges.get("circularity", (0.0, 1.0)))
        d_rng = tuple(initial_ranges.get("equivalent_diameter", (0, 200)))

        @magicgui(
            auto_call=True,
            # IMPORTANT: use FloatRangeSlider for floats; RangeSlider remains for ints
            circularity={"widget_type": "FloatRangeSlider", "min": 0.0, "max": 1.0, "step": 0.1, "value": c_rng, "label": "circularity"},
            diameter={"widget_type": "RangeSlider", "min": 0, "max": 200, "step": 10, "value": d_rng, "label": "equivalent_diameter"},
        )    
        def filter_points(circularity, diameter):
            circ = all_properties.get("circularity")
            diam = all_properties.get("equivalent_diameter")
            if circ is None or diam is None:
                print("[WARNING] Missing properties for filter.")
                return

            cmin, cmax = circularity
            dmin, dmax = diameter
            mask = (
                (circ >= cmin) & (circ <= cmax) &
                (diam >= dmin) & (diam <= dmax)
            )

            idx = np.nonzero(mask)[0]
            points.data         = all_coords_np[mask]
            points.text.values  = [texts_list[i] for i in idx]
            points.properties   = {k: v[mask] for k, v in all_properties.items()}

            # NEW: bubble up the current ranges so the widget can remember them
            if callable(on_filter_change):
                on_filter_change({
                    "circularity": (float(cmin), float(cmax)),
                    "equivalent_diameter": (float(dmin), float(dmax)),
                })

        try:
            for dw in list(getattr(viewer.window, "dock_widgets", [])):
                nm = getattr(dw, "name", None) or getattr(dw, "windowTitle", lambda: None)()
                if nm == "Filter Cells":
                    viewer.window.remove_dock_widget(dw)
        except Exception:
            pass
        viewer.window.add_dock_widget(filter_points.native, area="right", name="Filter Cells")

    # ---------- boxplot (tags from CSV) ----------
    if show_boxplot:
        _remove_dock("Intensity Boxplot")
        _remove_dock("Boxplot Controls")

        fig, ax = plt.subplots()
        canvas = FigureCanvas(fig)
        panel = QWidget()
        _panel_layout = QVBoxLayout(panel)
        _panel_layout.setContentsMargins(6, 6, 6, 6)
        _panel_layout.addWidget(canvas)

        mean_intensity_metrics = [k for k in all_properties if k.endswith("_mean_intensity")]
        default_metric = (
            "Actin_mean_intensity"
            if "Actin_mean_intensity" in mean_intensity_metrics
            else (mean_intensity_metrics[0] if mean_intensity_metrics else None)
        )

        def _plot_box(metric: str | None):
            ax.clear()
            if metric is None or metric not in points.properties:
                ax.set_title("No mean-intensity metric found")
                canvas.draw_idle()
                return

            coords = points.data
            if coords.size == 0:
                ax.set_title("No points after filtering")
                canvas.draw_idle()
                return

            frame_idx = coords[:, 0].astype(int)
            intens = np.asarray(points.properties[metric])
            groups = [tags_valid[i] for i in frame_idx]

            df_plot = pd.DataFrame({"Group": groups, metric: intens})
            buckets, labels = [], []
            for tag, g in df_plot.groupby("Group", sort=True):
                buckets.append(g[metric].values)
                labels.append(str(tag))

            if len(buckets) == 0:
                ax.set_title("No data to plot")
                canvas.draw_idle()
                return

            ax.boxplot(buckets, labels=labels, showfliers=True)
            ax.set_xlabel("Group")
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} by Group")
            for lab in ax.get_xticklabels():
                lab.set_rotation(20)
                lab.set_ha("right")
            fig.tight_layout()
            canvas.draw_idle()

        @magicgui(auto_call=True, metric={"label": "Metric", "choices": mean_intensity_metrics, "value": default_metric})
        def boxplot_controls(metric: str = default_metric):
            _plot_box(metric)

        def _on_points_changed(event=None):
            cur = boxplot_controls.metric.value if hasattr(boxplot_controls, "metric") else default_metric
            _plot_box(cur)

        points.events.data.connect(_on_points_changed)
        _plot_box(default_metric)

        viewer.window.add_dock_widget(panel, area="right", name="Intensity Boxplot")
        viewer.window.add_dock_widget(boxplot_controls, area="right", name="Boxplot Controls")

    # ---------- export UI ----------
    if export_csv:
        default_export_path = Path(output_folder) / "filtered_points_export.csv"
        default_export_path.parent.mkdir(parents=True, exist_ok=True)

        save_path_picker = FileEdit(label="Save CSV", mode="w", value=str(default_export_path))
        export_button = PushButton(label="Export CSV")

        def _do_export():
            if points.data.shape[0] == 0:
                print("[INFO] No points to export.")
                return

            out_path = str(save_path_picker.value)
            if not out_path.endswith(".csv"):
                out_path += ".csv"

            filtered_coords = points.data
            filtered_props  = points.properties

            data = {
                "frame_index": filtered_coords[:, 0],
                "y": filtered_coords[:, 1],
                "x": filtered_coords[:, 2],
            }
            for prop_name, prop_vals in filtered_props.items():
                data[prop_name] = prop_vals

            data["frame_name"] = [names_valid[int(i)] for i in filtered_coords[:, 0]]
            data["frame_tag"]  = [tags_valid[int(i)]  for i in filtered_coords[:, 0]]

            pd.DataFrame(data).to_csv(out_path, index=False)
            viewer.status = f"✅ Exported filtered points to: {out_path}"

        export_button.changed.connect(_do_export)
        export_widget = Container(widgets=[save_path_picker, export_button])
        viewer.window.add_dock_widget(export_widget, area="right", name="Export Filtered Points")
