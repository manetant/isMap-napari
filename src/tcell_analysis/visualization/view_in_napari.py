# --- paste the whole function below, keep your imports at top ---
from pathlib import Path
import numpy as np
import pandas as pd
import re as _re
from tifffile import imread
from tifffile import imread as _tiffread
#import dask.array as daQtWebEngineWidgets
import dask.array as da
from dask import delayed
from magicgui import magicgui
from magicgui.widgets import Container, FileEdit, PushButton
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from qtpy.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QLabel, QSlider
from qtpy.QtWidgets import QPushButton, QFileDialog, QHBoxLayout, QSizePolicy
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
#from PyQt5.QtWebEngineWidgets import QWebEngineView
#import plotly.graph_objects as go
from matplotlib import cm
from matplotlib.figure import Figure as _Fig
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

import re


def reset_tcell_session(viewer):
    """Remove all layers/docks/figures we added and clear callbacks & refs."""
    PLUGIN_TAG = "tcell_analysis_layers"

    # 1) Remove layers we created
    to_rm = []
    for layer in list(viewer.layers):
        md = getattr(layer, "metadata", {}) or {}
        if md.get(PLUGIN_TAG):
            to_rm.append(layer)
        # also sweep common names in case tag was lost
        base = layer.name.split(" [", 1)[0]
        if base in ("RGB Composite", "Mask Stack", "Cell Labels"):
            to_rm.append(layer)
    for l in set(to_rm):
        try:
            viewer.layers.remove(l)
        except Exception:
            pass

    # 2) Remove our dock widgets (by name or by handle if present)
    def _rm_dock(nm):
        try:
            viewer.window.remove_dock_widget(nm)
            return
        except Exception:
            pass
        try:
            d = getattr(viewer.window, "_wrapped_dock_widgets", {})
            if nm in d:
                viewer.window.remove_dock_widget(d[nm])
        except Exception:
            pass

    for nm in [
        "Analysis Plots",
        "Filter Cells",
        "Radial Condition Viewer",
        "Radial Line Controls",
        "Radial Line Profiles",
        "Intensity Boxplot",
        "Boxplot Controls",
        "PCC (metrics)",
        #"isMap Progress",
    ]:
        _rm_dock(nm)

    # 3) Disconnect any event callbacks we registered on the last run
    try:
        for cb in getattr(viewer, "_tcell_points_callbacks", []):
            try:
                pts = getattr(viewer, "_tcell_points_layer", None)
                if pts is not None:
                    pts.events.data.disconnect(cb)
            except Exception:
                pass
    except Exception:
        pass
    setattr(viewer, "_tcell_points_callbacks", [])
    setattr(viewer, "_tcell_points_layer", None)

    # 4) Close any Matplotlib figures we kept references to
    try:
        for obj in list(getattr(viewer, "_tcell_refs", []) or []):
            try:
                if hasattr(obj, "figure"):           # e.g., a Canvas
                    import matplotlib.pyplot as _plt
                    _plt.close(obj.figure)
                elif hasattr(obj, "canvas") and hasattr(obj.canvas, "figure"):
                    import matplotlib.pyplot as _plt
                    _plt.close(obj.canvas.figure)
                else:
                    # direct Figure
                    
                    if isinstance(obj, _Fig):
                        import matplotlib.pyplot as _plt
                        _plt.close(obj)
            except Exception:
                pass
    except Exception:
        pass
    setattr(viewer, "_tcell_refs", [])

    # 5) Clear overlay/status
    try:
        viewer.text_overlay.text = ""
        viewer.text_overlay.visible = False
    except Exception:
        pass
    try:
        viewer.status = "Session reset."
    except Exception:
        pass


def show_analysis_results(
    viewer,
    output_folder,
    channel_names,
    tasks_with_tags,       
    text_feature=None,
    text_size=12,
    text_color="darkgreen",
    rgb=False,
    initial_ranges=None,         
    on_filter_change=None, 
    show_filter: bool = True,
    show_boxplot: bool = True,
    show_pcc: bool = True,
    show_radial_viewer: bool = True,
    um_per_px: float | None = None,
):
    if initial_ranges is None:
        initial_ranges = {}

    try:
        reset_tcell_session(viewer)
    except Exception:
        # don't block; continue even if cleanup hits a harmless error
        pass
    try:
        um_per_px = float(um_per_px) if um_per_px is not None else None
    except Exception:
        um_per_px = None

    # Tag we place on all layers we add so we can clean them up later
    PLUGIN_TAG = "tcell_analysis_layers"
    radial_controls = None
    radial_line_profiles = None
    panel_lp = None          # matplotlib canvas panel for radial profiles
    boxplot_controls = None
    panel = None             # matplotlib canvas panel for boxplot
    # PCC (metrics) tab placeholders
    pcc_controls = None
    panel_pcc_metrics = None


    # keep strong refs for dock widgets / magicgui objects
    def _keep_refs(*objs):
        try:
            bucket = getattr(viewer, "_tcell_refs", None)
            if bucket is None:
                bucket = []
                setattr(viewer, "_tcell_refs", bucket)
            bucket.extend(objs)
        except Exception:
            pass

    def _base(nm: str) -> str:
        # strip napari's auto-suffix like "ICAM1 [1]"
        m = re.match(r"^(.*) \[\d+\]$", nm)
        return m.group(1) if m else nm

    def _add_single_colorbar(fig, ax, im):
        """Ensure exactly one colorbar is visible for the given axes."""
        # Remove old colorbars
        for a in list(fig.axes):
            if a is not ax:
                try:
                    fig.delaxes(a)
                except Exception:
                    pass

        # Add new colorbar with white text for dark theme
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.04)
        cbar = fig.colorbar(im, cax=cax)

        # Style for dark background
        cbar.ax.set_ylabel("Intensity", rotation=270, labelpad=12, color="white")
        cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")
        for spine in cbar.ax.spines.values():
            spine.set_edgecolor("white")

        return cbar

    def _purge_old_layers():
        # names we might create in this function
        expected = set(["RGB Composite", "Mask Stack", "Cell Labels"])
        expected.update(ch_names)  # ICAM1, pTyr, Actin, etc.

        to_remove = []
        for layer in list(viewer.layers):
            # 1) remove anything we previously added (by metadata tag)
            if getattr(layer, "metadata", None) and layer.metadata.get(PLUGIN_TAG):
                to_remove.append(layer)
                continue
            # 2) also remove layers whose *base* name matches what we are about to add
            if _base(layer.name) in expected:
                to_remove.append(layer)

        for l in to_remove:
            try:
                viewer.layers.remove(l)
            except Exception:
                pass
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
            d = getattr(viewer.window, "_wrapped_dock_widgets", {})
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
        name = frame_dir.name
        p = frame_dir / f"{name}.tiff"
        if p.exists():
            return p
        cands = [
            q for q in frame_dir.glob("*.tif*")
            if not q.name.lower().endswith("_mask.tiff") and "mask" not in q.name.lower()
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
        """
        Load mask + per_cell_features.csv and return (mask, df_with_index, tag, seg_name).
        seg_name is inferred from "<SEG>_mask.tiff" inside fd.
        """
        # find any "*_mask.tiff" in the frame directory
        mask_candidates = sorted([p for p in fd.glob("*_mask.tiff")])
        if not mask_candidates:
            raise FileNotFoundError(fd / "*_mask.tiff")
        mask_path = mask_candidates[0]

        csv_path = fd / "per_cell_features.csv"
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

        # infer segmentation channel from file name "<SEG>_mask.tiff"
        seg_name = mask_path.name[:-len("_mask.tiff")] if mask_path.name.lower().endswith("_mask.tiff") else "N/A"
        return mask, df, tag, seg_name



    # ---------- discover frames ----------
    # discover frames by masks
    root = Path(output_folder)
    mask_hits = sorted(root.rglob("*_mask.tiff"))
    if not mask_hits:
        raise ValueError(f"No *_mask.tiff under {root}")

    frame_dirs = sorted({p.parent for p in mask_hits})

    # Load lightweight things eagerly (mask/CSV/tag) and collect image paths
    image_paths, masks, dfs, frame_tags, seg_names = [], [], [], [], []
    for fd in frame_dirs:
        try:
            mask, df, tag, seg_name = load_mask_and_df(fd)
        except (FileNotFoundError, ValueError) as e:
            print(f"[WARN] Skipping {fd}: {e}")
            continue
        masks.append(mask)
        dfs.append(df)
        frame_tags.append(tag)
        seg_names.append(seg_name)
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

    _purge_old_layers()

    if rgb and num_channels >= 3:
        lows  = [clims[i][0] for i in range(min(3, len(clims)))]
        highs = [clims[i][1] for i in range(min(3, len(clims)))]
        viewer.add_image(
            image_stack[..., :3],
            name="RGB Composite",
            rgb=True,
            opacity=1.0,
            contrast_limits=(min(lows), max(highs)) if clims else None,
            metadata={PLUGIN_TAG: True},
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
            metadata={"channel_index": c, PLUGIN_TAG: True},
        )


    # ---------- labels stack ----------
    # Only include masks for valid image frames (to keep leading axis aligned)
    masks_valid = [masks[i] for i in valid_idxs]
    mask_stack = np.stack(masks_valid, axis=0)
    if not np.issubdtype(mask_stack.dtype, np.integer):
        mask_stack = mask_stack.astype(np.int32, copy=False)
    #viewer.add_labels(mask_stack, name="Mask Stack", metadata={PLUGIN_TAG: True})
    labels_layer = viewer.add_labels(
        mask_stack,
        name="Mask Stack",
        metadata={PLUGIN_TAG: True},
    )

    # show only the borders (in pixels). 1–2 looks nice; 0 = filled
    labels_layer.contour = 3
    labels_layer.opacity = 1.0         # keep lines fully visible
    labels_layer.blending = "translucent"  # overlays cleanly on grayscale channels

    # ---------- points/properties/text (vectorized) ----------
    # Align dfs and tags to valid_idxs too
    dfs_valid       = [dfs[i] for i in valid_idxs]
    tags_valid      = [frame_tags[i] for i in valid_idxs]
    names_valid     = [frame_dirs[i].name for i in valid_idxs]
    seg_names_valid = [seg_names[i] for i in valid_idxs]


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
        face_color="transparent",
        properties=all_properties,
        metadata={PLUGIN_TAG: True},
    )

    points.edge_color = "red"
    points.text = {"string": texts_list, 
                   "size": text_size, 
                   "color": text_color,
                   "bold": True,
                   "anchor": "center"}

    # add condition per point as a property so we can filter PCC by it
    #_cond_per_point = np.array([tags_valid[int(fi)] for fi in all_coords_np[:, 0]], dtype=object)
    #points.properties["frame_tag"] = _cond_per_point

    # ---------- overlay using CSV tag ----------
    viewer.text_overlay.visible = True
    viewer.text_overlay.position = "top_left"

    def _update_overlay(event=None):
        if viewer.dims.ndim and viewer.dims.current_step:
            idx = int(viewer.dims.current_step[0])
        else:
            idx = 0
        idx = max(0, min(idx, len(names_valid) - 1))
        n_cells = len(dfs_valid[idx])
        seg_name = seg_names_valid[idx] if idx < len(seg_names_valid) else "N/A"
        viewer.text_overlay.text = (
            f"Frame: {names_valid[idx]} | Group: {tags_valid[idx]} | Cells: {n_cells} | Seg: {seg_name}"
        )
        viewer.text_overlay.font_size = 16

    viewer.dims.events.current_step.connect(_update_overlay)
    _update_overlay()

    # ---------- Radial Condition Viewer (optional) ----------
    if show_radial_viewer:
        # find available radial channels by scanning the cond dir of each condition
        cond_dirs = {}
        root_path = Path(output_folder)
        for tag in sorted(set(tags_valid)):
            top = root_path / f"res_{tag}"
            if top.exists():
                cond_dirs[tag] = top
            else:
                hits = list(root_path.rglob(f"res_{tag}"))
                cond_dirs[tag] = hits[0] if hits else top

        # build channel list: start with ALL image channels
        radial_channels = list(ch_names)

        for tag, cdir in cond_dirs.items():
            if not cdir.exists():
                continue
            for p in sorted(cdir.glob("*_radTotAv_scaled.tif")) + sorted(cdir.glob("*_radTotAv_padded.tif")):
                nm = p.name
                ch = (nm
                    .replace("_radTotAv_scaled.tif", "")
                    .replace("_radTotAv_padded.tif", ""))
                if ch not in radial_channels:
                    radial_channels.append(ch)

        def _upsert_image(name, data):
            try:
                layer = viewer.layers[name]
                layer.data = data
                layer.metadata["tcell_analysis_layers"] = True
                return layer
            except KeyError:
                return viewer.add_image(
                    data, name=name,
                    metadata={"tcell_analysis_layers": True},
                    colormap="gray"
                )

        def _radial_path(cdir: Path, channel: str, kind: str, scaled: bool) -> Path | None:
            base = {"Total Average": "radTotAv", "Montage": "radMontage", "Stack": "radStack"}[kind]
            suff = "_scaled" if scaled else "_padded"
            p = cdir / f"{channel}_{base}{suff}.tif"
            return p if p.exists() else None

        # --- inside show_analysis_results ---
        fig_radimg, ax_radimg = plt.subplots()

        # make the figure dark-themed like Plotly
        fig_radimg.patch.set_facecolor("black")
        ax_radimg.set_facecolor("black")
        ax_radimg.tick_params(colors="white")
        ax_radimg.title.set_color("white")

        canvas_radimg = FigureCanvas(fig_radimg)
        slider_z = QSlider(Qt.Horizontal)
        slider_z.setMinimum(0)
        slider_z.hide()  # shown only for 3D stacks

        RADIAL_CMAPS = ["gray", "magma", "inferno", "viridis"]

        # Add a magicgui control for colormap selection
        @magicgui(
            auto_call=True,
            cmap={"choices": RADIAL_CMAPS, "label": "Colormap"},
        )
        def radial_colormap_control(cmap: str = "gray"):
            """Dummy control just to pick the active colormap."""
            return cmap

        # --- Automatically refresh radial image when colormap changes ---
        def _on_cmap_change(event=None):
            try:
                vals = radial_controls
                radial_controls(
                    channel=vals.channel.value,
                    kind=vals.kind.value,
                    tag=vals.tag.value,
                    scaled=vals.scaled.value,
                )
            except Exception as e:
                viewer.status = f"⚠️ Colormap update failed: {e}"

        radial_colormap_control.changed.connect(_on_cmap_change)

        def _show_radial_in_panel(path: Path, title: str):
            arr = _tiffread(str(path))

            for a in list(fig_radimg.axes):
                if a is not ax_radimg:
                    try:
                        fig_radimg.delaxes(a)
                    except Exception:
                        pass            
            ax_radimg.clear()
            cmap = radial_colormap_control.cmap.value

            if arr.ndim == 3:  # 3D stack (Z, Y, X)
                slider_z.show()
                slider_z.setMaximum(arr.shape[0] - 1)
                mid = arr.shape[0] // 2
                slider_z.setValue(mid)

                img2d = arr[mid, :, :]
                im = ax_radimg.imshow(img2d, cmap=cmap)
                ax_radimg.set_title(f"{title} — Slice {mid+1}/{arr.shape[0]}", 
                                    color="white")
                ax_radimg.axis("off")
                _add_single_colorbar(fig_radimg, ax_radimg, im)
                fig_radimg.tight_layout()
                canvas_radimg.draw_idle()

                def _update_slice(val):
                    ax_radimg.clear()
                    im = ax_radimg.imshow(arr[val, :, :], cmap=cmap)
                    ax_radimg.set_title(f"{title} — Slice {val+1}/{arr.shape[0]}",
                                        color="white")
                    ax_radimg.axis("off")
                    _add_single_colorbar(fig_radimg, ax_radimg, im)
                    fig_radimg.tight_layout()
                    canvas_radimg.draw_idle()

                try:
                    slider_z.valueChanged.disconnect()
                except Exception:
                    pass
                slider_z.valueChanged.connect(_update_slice)

            elif arr.ndim == 2:
                slider_z.hide()
                im = ax_radimg.imshow(arr, cmap=cmap)
                ax_radimg.set_title(title, color="white")
                ax_radimg.axis("off")
                _add_single_colorbar(fig_radimg, ax_radimg, im)
                fig_radimg.tight_layout()
                canvas_radimg.draw_idle()

            else:
                slider_z.hide()
                viewer.status = f"⚠️ Unexpected array shape {arr.shape} for {path.name}"

        @magicgui(
            auto_call=True,
            channel={"choices": radial_channels or ["(none)"], "label": "Channel"},
            kind={"choices": ["Total Average", "Montage", "Stack"], "label": "Show"},
            tag={"choices": sorted(set(tags_valid)), "label": "Condition"},
            scaled={"widget_type": "CheckBox", "label": "Scaled (resize to target)", "value": True},
        )
        def _radial_controls_impl(
            channel: str = radial_channels[0] if radial_channels else "(none)",
            kind: str = "Total Average",
            tag: str = sorted(set(tags_valid))[0] if tags_valid else "N/A",
            scaled: bool = True,
        ):
            if channel == "(none)":
                return
            cdir = cond_dirs.get(tag, None)
            if not cdir or not cdir.exists():
                return

            path = _radial_path(cdir, channel, kind, scaled)
            if not path:
                viewer.status = f"⚠️ Missing radial ({'scaled' if scaled else 'padded'}) for {channel}/{kind}"
                return

            title = f"{channel} — {kind} — {tag} — {'scaled' if scaled else 'padded'}"
            _show_radial_in_panel(path, title)

        radial_controls = _radial_controls_impl 
        radial_colormap_control.cmap.changed.connect(lambda *_: radial_controls(
            channel=radial_controls.channel.value,
            kind=radial_controls.kind.value,
            tag=radial_controls.tag.value,
            scaled=radial_controls.scaled.value,
        ))

        # Automatically display the first available radial image on load
        try:
            first_ch = radial_channels[0] if radial_channels else None
            first_tag = sorted(set(tags_valid))[0] if tags_valid else None
            if first_ch and first_tag:
                radial_controls(
                    channel=first_ch,
                    kind="Total Average",
                    tag=first_tag,
                    scaled=True
                )
        except Exception as e:
            viewer.status = f"⚠️ Could not init Radial Image display: {e}"            

        try:
            # ==== Radial Line Profiles (MFI normalized along a diagonal) ====
            # ==== Radial Line Profiles (Matplotlib, cross-platform) ====

            def _diagonal_profile(img2d: np.ndarray, diagonal: str = "main") -> np.ndarray:
                if img2d.ndim != 2:
                    raise ValueError("Expected 2D image for radial profile")
                if diagonal == "anti":
                    img2d = np.fliplr(img2d)

                n = min(img2d.shape)
                y0 = (img2d.shape[0] - n) // 2
                x0 = (img2d.shape[1] - n) // 2
                sub = img2d[y0:y0+n, x0:x0+n]

                prof = np.diag(sub).astype(np.float64, copy=False)
                m = np.nanmax(prof)
                return prof / m if m > 0 else prof


            # ---- Figure ----
            fig_lp, ax_lp = plt.subplots(figsize=(6, 4))
            fig_lp.patch.set_facecolor("#111111")
            ax_lp.set_facecolor("#111111")

            ax_lp.tick_params(
                axis="both",
                colors="white",
                labelsize=9,
            )

            for spine in ax_lp.spines.values():
                spine.set_color("white")

            ax_lp.xaxis.label.set_color("white")
            ax_lp.yaxis.label.set_color("white")
            ax_lp.title.set_color("white")

            ax_lp.grid(True, color="white", alpha=0.15, linewidth=0.5)

            canvas_lp = FigureCanvas(fig_lp)


            # ---- Controls ----
            all_tags = sorted(set(tags_valid))
            all_channels = sorted(radial_channels)

            @magicgui(
                auto_call=True,
                tag={"choices": all_tags, "label": "Condition"},
                x_mode={"choices": ["Normalized (-1..1)", "Micrometers (µm)"], "label": "X-axis"},
                scaled={"widget_type": "CheckBox", "label": "Scaled", "value": True},
            )
            def radial_line_profiles(
                tag: str = all_tags[0],
                x_mode: str = "Normalized (-1..1)",
                scaled: bool = True,
            ):
                ax_lp.clear()
                ax_lp.set_facecolor("black")

                plotted = False

                for ch in all_channels:
                    cdir = cond_dirs.get(tag)
                    if not cdir or not cdir.exists():
                        continue

                    path = _radial_path(cdir, ch, "Total Average", scaled)
                    if not path:
                        continue

                    arr = _tiffread(str(path))
                    prof = _diagonal_profile(arr)
                    if prof.size == 0:
                        continue

                    if x_mode == "Micrometers (µm)" and um_per_px:
                        half = (len(prof) - 1) / 2
                        x = (np.arange(len(prof)) - half) * um_per_px
                        xlabel = "Distance (µm)"
                    else:
                        x = np.linspace(-1, 1, len(prof))
                        xlabel = "Distance (normalized)"

                    ax_lp.plot(x, prof, label=ch)
                    plotted = True

                if not plotted:
                    ax_lp.set_title("No radial profiles available", color="white")
                else:
                    ax_lp.set_title(f"Radial Profiles — {tag}", color="white")
                    ax_lp.set_xlabel(xlabel)
                    ax_lp.set_ylabel("Normalized MFI")
                    ax_lp.legend(fontsize=8, facecolor="black", labelcolor="white")

                fig_lp.tight_layout()
                canvas_lp.draw_idle()


            # ---- Export buttons ----
            btn_row = QHBoxLayout()
            btn_save_png = QPushButton("Save PNG")
            btn_save_csv = QPushButton("Export CSV")
            btn_row.addWidget(btn_save_png)
            btn_row.addWidget(btn_save_csv)

            # ---- Force initial radial profile render ----
            try:
                if all_tags:
                    radial_line_profiles(
                        tag=all_tags[0],
                        x_mode="Normalized (-1..1)",
                        scaled=True,
                    )
            except Exception as e:
                viewer.status = f"⚠️ Radial profile init failed: {e}"

            def _save_radial_png():
                out = Path(output_folder) / f"radial_profiles_{radial_line_profiles.tag.value}.png"
                fig_lp.savefig(out, dpi=300, bbox_inches="tight")
                viewer.status = f"Saved: {out}"


            def _export_radial_csv():
                rows = []
                tag = radial_line_profiles.tag.value
                scaled = radial_line_profiles.scaled.value

                for ch in all_channels:
                    cdir = cond_dirs.get(tag)
                    if not cdir:
                        continue

                    path = _radial_path(cdir, ch, "Total Average", scaled)
                    if not path:
                        continue

                    arr = _tiffread(str(path))
                    prof = _diagonal_profile(arr)
                    if prof.size == 0:
                        continue

                    for i, v in enumerate(prof):
                        rows.append({
                            "condition": tag,
                            "channel": ch,
                            "index": i,
                            "mfi_norm": float(v),
                        })

                if not rows:
                    viewer.status = "No data to export"
                    return

                df = pd.DataFrame(rows)
                out = Path(output_folder) / f"radial_profiles_{tag}.csv"
                df.to_csv(out, index=False)
                viewer.status = f"Saved: {out}"


            btn_save_png.clicked.connect(_save_radial_png)
            btn_save_csv.clicked.connect(_export_radial_csv)


            # ---- Panel ----
            panel_lp = QWidget()
            layout_lp = QVBoxLayout(panel_lp)
            layout_lp.setContentsMargins(6, 6, 6, 6)
            layout_lp.addWidget(radial_line_profiles.native)
            layout_lp.addWidget(canvas_lp)
            layout_lp.addLayout(btn_row)

            _keep_refs(fig_lp, ax_lp, canvas_lp, panel_lp, btn_save_png, btn_save_csv)

        except Exception as e:
            viewer.status = f"⚠️ Radial Line Profiles disabled: {e}"


    
    else:
        _remove_dock("Radial Condition Viewer")   # hide during segmentation/pilot

    # ---------- filtering widget (range sliders are nicer, but keep your sliders) ----------
    # defaults if not provided
    if show_filter:
        # Pull full distributions once (all cells, unfiltered)
        circ_all = np.asarray(all_properties.get("circularity", np.array([])), dtype=float)
        diam_all = np.asarray(all_properties.get("equivalent_diameter", np.array([])), dtype=float)

        # ---- dynamic diameter max ----
        if diam_all.size > 0:
            diam_max_val = float(np.nanmax(diam_all))
            if not np.isfinite(diam_max_val) or diam_max_val <= 0:
                diam_max_val = 200.0
        else:
            diam_max_val = 200.0

        # round to nice number
        diam_max_px = int(np.ceil(diam_max_val / 10.0) * 10)

        # circularity range (static)
        c_rng = tuple(initial_ranges.get("circularity", (0.0, 1.0)))

        # diameter range (dynamic default)
        if "equivalent_diameter" in initial_ranges:
            d_rng = tuple(initial_ranges["equivalent_diameter"])
        else:
            d_rng = (0, diam_max_px)

        @magicgui(
            auto_call=True,
            circularity={
                "widget_type": "FloatRangeSlider",
                "min": 0.0,
                "max": 1.0,
                "step": 0.1,
                "value": c_rng,
                "label": "Circularity",
            },
            diameter={
                "widget_type": "RangeSlider",
                "min": 0,
                "max": diam_max_px,
                "step": 10,
                "value": d_rng,
                "label": "Eq. Diam (px)",
            },
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

        # ---- Histograms (styled for dark/transparent background) ----
        plt_style = {
            "facecolor": (0, 0, 0, 0),        # transparent background
            "edgecolor": "none"
        }

        # Circularity
        fig_circ, ax_circ = plt.subplots(figsize=(4.6, 2.2))
        fig_circ.patch.set_alpha(0.0)           # transparent figure
        ax_circ.set_facecolor("none")           # transparent axes background
        ax_circ.tick_params(colors="white")     # white ticks for dark theme
        ax_circ.title.set_color("white")
        ax_circ.xaxis.label.set_color("white")
        ax_circ.yaxis.label.set_color("white")

        canvas_circ = FigureCanvas(fig_circ)
        if circ_all.size:
            ax_circ.hist(
                circ_all[np.isfinite(circ_all)],
                bins=20,
                color="#4F9DF7",
                alpha=0.6,
                edgecolor="white"
            )
            #ax_circ.set_title("Circularity (all cells)")
            ax_circ.set_xlabel("Circularity")
            ax_circ.set_ylabel("Count")
            ax_circ.axvline(c_rng[0], ls="--", color="white", alpha=0.6)
            ax_circ.axvline(c_rng[1], ls="--", color="white", alpha=0.6)
        else:
            ax_circ.set_title("Circularity: no data")
        fig_circ.tight_layout()
        canvas_circ.draw_idle()

        # Diameter
        fig_diam, ax_diam = plt.subplots(figsize=(4.6, 2.2))
        fig_diam.patch.set_alpha(0.0)
        ax_diam.set_facecolor("none")
        ax_diam.tick_params(colors="white")
        ax_diam.title.set_color("white")
        ax_diam.xaxis.label.set_color("white")
        ax_diam.yaxis.label.set_color("white")

        canvas_diam = FigureCanvas(fig_diam)
        if diam_all.size:
            ax_diam.hist(
                diam_all[np.isfinite(diam_all)],
                bins=20,
                color="#4F9DF7",
                alpha=0.6,
                edgecolor="white"
            )
            #ax_diam.set_title("Equivalent diameter (all cells)")
            ax_diam.set_xlabel("Equivalent diameter (px)")
            ax_diam.set_ylabel("Count")
            ax_diam.axvline(d_rng[0], ls="--", color="white", alpha=0.6)
            ax_diam.axvline(d_rng[1], ls="--", color="white", alpha=0.6)
        else:
            ax_diam.set_title("Equivalent diameter: no data")
        fig_diam.tight_layout()
        canvas_diam.draw_idle()


        # ---- Compose a panel: sliders first, histograms below ----
        _remove_dock("Filter Cells")
        filter_panel = QWidget()
        _filter_layout = QVBoxLayout(filter_panel)
        _filter_layout.setContentsMargins(6, 6, 6, 6)

        lbl_hint = QLabel("Below are distributions of all cells. Dashed lines mark initial ranges.")
        lbl_hint.setWordWrap(True)

        _filter_layout.addWidget(filter_points.native)  # sliders on top
        _filter_layout.addWidget(lbl_hint)
        _filter_layout.addWidget(canvas_circ)
        _filter_layout.addWidget(canvas_diam)

        viewer.window.add_dock_widget(filter_panel, area="right", name="Filter Cells")

        # ---- Make the Filter Cells dock larger by default ----
        try:
            # First, give the inner panel a minimum size (so the canvas + sliders fit)
            filter_panel.setMinimumHeight(600)
            filter_panel.setMinimumWidth(450)
            filter_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

            # Then try to resize the actual dock widget if accessible
            docks = getattr(viewer.window, "dock_widgets", {})
            dock_widget = docks.get("Filter Cells", None)
            if dock_widget is not None:
                dock_widget.setMinimumHeight(600)
                dock_widget.setMinimumWidth(450)
                dock_widget.resize(450, 600)
        except Exception as e:
            viewer.status = f"⚠️ Could not resize Filter Cells dock: {e}"

        # prevent GC
        _keep_refs(
            fig_circ, ax_circ, canvas_circ,
            fig_diam, ax_diam, canvas_diam,
            filter_panel, lbl_hint, filter_points
        )

    else:
        _remove_dock("Filter Cells")

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

        # --- Save MFI boxplot as PNG ---
        row_png = QHBoxLayout()
        btn_save_mfi_png = QPushButton("Save plot")
        row_png.addWidget(btn_save_mfi_png)
        _panel_layout.addLayout(row_png)

        def _save_mfi_png():
            try:
                # use current UI state for filename (metric + normalization flag)
                cur_m = boxplot_controls.metric.value if hasattr(boxplot_controls, "metric") else "metric"
                cur_n = bool(boxplot_controls.normalized.value) if hasattr(boxplot_controls, "normalized") else False
                suffix = "_normalized" if cur_n else ""
                default_png = str((Path(output_folder) / f"mfi_{cur_m}{suffix}.png"))

                parent = getattr(viewer.window, "_qt_window", None)
                out, _ = QFileDialog.getSaveFileName(parent, "Save MFI plot as PNG", default_png, "PNG (*.png)")
                if not out:
                    return
                fig.savefig(out, dpi=300, bbox_inches="tight")
                viewer.status = f"Saved MFI PNG: {out}"
            except Exception as e:
                viewer.status = f"⚠️ Save MFI PNG failed: {e}"

        btn_save_mfi_png.clicked.connect(_save_mfi_png)

        # keep refs so Qt/GC doesn't eat the button/closure
        _keep_refs(btn_save_mfi_png)

        # choose default metric using the detected seg name of the first valid frame
        mean_intensity_metrics = [k for k in all_properties if k.endswith("_mean_intensity")]
        default_metric = None
        if mean_intensity_metrics:
            if seg_names_valid and seg_names_valid[0]:
                cand = f"{seg_names_valid[0]}_mean_intensity"
                default_metric = cand if cand in mean_intensity_metrics else None
            if default_metric is None:
                default_metric = mean_intensity_metrics[0]

        def _pick_series(metric: str, normalized: bool) -> tuple[str, np.ndarray] | None:
            """
            Return (label, data_vector) taken from points.properties for 'metric'.
            If normalized=True and a '<metric>_norm' column exists, use it; else None.
            """
            if not metric:
                return None
            if not normalized:
                vals = points.properties.get(metric, None)
                if vals is None:
                    return None
                return (metric, np.asarray(vals))
            
            # try normalized version
            norm_col = f"{metric}_bg_norm"
            vals = points.properties.get(norm_col, None)
            if vals is None:
                return None
            return (norm_col, np.asarray(vals))

        def _plot_box(metric: str | None, normalized: bool):
            ax.clear()
            if metric is None or (metric not in points.properties and f"{metric}_norm" not in points.properties):
                ax.set_title("No mean-intensity metric found")
                canvas.draw_idle()
                return

            coords = points.data
            if coords.size == 0:
                ax.set_title("No points after filtering")
                canvas.draw_idle()
                return
            
            picked = _pick_series(metric, normalized)
            if picked is None:
                if normalized:
                    ax.set_title(f"No normalized data for '{metric}'")
                else:
                    ax.set_title(f"No data for '{metric}'")
                canvas.draw_idle()
                return
            
            colname, intens_all = picked
            frame_idx = coords[:, 0].astype(int)
            groups = [tags_valid[i] for i in frame_idx]

            df_plot = pd.DataFrame({"Group": groups, colname: intens_all})
            buckets, labels = [], []
            for tag, g in df_plot.groupby("Group", sort=True):
                # keep only finite
                v = pd.to_numeric(g[colname], errors="coerce").dropna().values
                if v.size == 0:
                    continue
                buckets.append(v)
                labels.append(str(tag))

            if len(buckets) == 0:
                ax.set_title("No data to plot")
                canvas.draw_idle()
                return

            ax.boxplot(
                buckets, 
                labels=labels,
                widths=0.6,
                patch_artist=True,
                boxprops=dict(facecolor="lightgray", edgecolor="black"),
                medianprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                capprops=dict(color="black"),
                showfliers=True)
            
            ax.set_xlabel("Condition")
            ax.set_ylabel(metric + (" (A.U.)" if not normalized else " (normalized)"))
            title = (metric if not normalized else f"{metric} (normalized)")
            ax.set_title(f"{title} by condition")
            for lab in ax.get_xticklabels():
                lab.set_rotation(20)
                lab.set_ha("right")
            fig.tight_layout()
            canvas.draw_idle()

        @magicgui(
            auto_call=True,
            metric={"label": "Metric", "choices": mean_intensity_metrics, "value": default_metric},
            normalized={"widget_type": "CheckBox", "label": "Show normalized", "value": False},
        )
        def boxplot_controls(metric: str = default_metric, normalized: bool = False):
            _plot_box(metric, normalized)

        def _on_points_changed(event=None):
            cur_m = boxplot_controls.metric.value if hasattr(boxplot_controls, "metric") else default_metric
            cur_n = bool(boxplot_controls.normalized.value) if hasattr(boxplot_controls, "normalized") else False
            _plot_box(cur_m, cur_n)

        points.events.data.connect(_on_points_changed)
        _plot_box(default_metric, False)  # default: raw (not normalized

    else:
        _remove_dock("Intensity Boxplot") 
        _remove_dock("Boxplot Controls")
    
    # ---------- PCC (metrics) tab ----------
    # Columns are named: PCC_{channelA}_vs_{channelB}
    if show_pcc:
        cond_choices = sorted(set(tags_valid))
        
        # ch_names is defined earlier from channel_names / image stack shape
        _all_channels_for_pcc = list(ch_names)

        # Helper to fetch PCC values for (ref=R, target=T)
        def _get_pcc_values(R: str, T: str, cond: str) -> np.ndarray | None:
            # build condition mask (all vs specific)
            if "tag" in points.properties:
                cond_all = np.asarray(points.properties["tag"])
                cond_mask = cond_all == cond
            else:
                # fallback: no condition column available
                cond_mask = np.ones(points.data.shape[0], dtype=bool)

            # Try both PCC column orders and apply condition mask
            for col in (f"PCC_{R}_vs_{T}", f"PCC_{T}_vs_{R}"):
                vals_all = points.properties.get(col, None)
                if vals_all is None:
                    continue
                vals = np.asarray(vals_all, dtype=float)
                vals = vals[cond_mask]
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    return vals
            return None


        if _all_channels_for_pcc:
            # Figure + canvas
            fig_pcc_m, ax_pcc_m = plt.subplots()
            canvas_pcc_m = FigureCanvas(fig_pcc_m)
            panel_pcc_metrics = QWidget()
            _pcc_layout = QVBoxLayout(panel_pcc_metrics)
            _pcc_layout.setContentsMargins(6, 6, 6, 6)
            _pcc_layout.addWidget(canvas_pcc_m)

            row_pcc_png = QHBoxLayout()
            btn_save_pcc_png = QPushButton("Save plot")
            row_pcc_png.addWidget(btn_save_pcc_png)
            _pcc_layout.addLayout(row_pcc_png)

            def _save_pcc_png():
                try:
                    cond = pcc_controls.condition.value if hasattr(pcc_controls, "condition") else "cond"
                    tgt  = pcc_controls.target.value if hasattr(pcc_controls, "target") else "target"
                    default_png = str((Path(output_folder) / f"pcc_{cond}_{tgt}.png"))
                    parent = getattr(viewer.window, "_qt_window", None)
                    out, _ = QFileDialog.getSaveFileName(parent, "Save PCC plot as PNG", default_png, "PNG (*.png)")
                    if not out:
                        return
                    fig_pcc_m.savefig(out, dpi=300, bbox_inches="tight")
                    viewer.status = f"Saved PCC PNG: {out}"
                except Exception as e:
                    viewer.status = f"⚠️ Save PCC PNG failed: {e}"

            btn_save_pcc_png.clicked.connect(_save_pcc_png)
            _keep_refs(btn_save_pcc_png)

            @magicgui(
                auto_call=True,
                condition={"choices": cond_choices, "label": "Condition"},
                target={"choices": _all_channels_for_pcc, "label": "Target channel (Y-axis)"},
                show_points={"widget_type": "CheckBox", "label": "Show points", "value": True},
                show_boxes={"widget_type": "CheckBox", "label": "Show boxes", "value": True},
            )
            def pcc_controls(
                condition: str = cond_choices[0],
                target: str = _all_channels_for_pcc[0] if _all_channels_for_pcc else "",
                show_points: bool = True,
                show_boxes: bool = True,
            ):
                ax_pcc_m.clear()

                if not target:
                    ax_pcc_m.set_title("Pick a target channel")
                    canvas_pcc_m.draw_idle()
                    return

                if points.data.size == 0:
                    ax_pcc_m.set_title("No points after filtering")
                    canvas_pcc_m.draw_idle()
                    return

                labels, data = [], []

                for R in _all_channels_for_pcc:
                    if R == target:
                        continue
                    vals = _get_pcc_values(R, target, condition)
                    if vals is None:
                        continue

                    vals = np.asarray(vals)
                    vals = vals[np.isfinite(vals)]
                    if vals.size == 0:
                        continue
                    labels.append(str(R))
                    data.append(vals)

                if len(data) == 0:
                    ax_pcc_m.set_title(f"No PCC columns for target: {target} (cond: {condition})")
                    canvas_pcc_m.draw_idle()
                    return

                xpos = np.arange(len(labels))
                if show_boxes:
                    ax_pcc_m.boxplot(data, positions=xpos, 
                                     patch_artist=True, 
                                     boxprops=dict(facecolor="lightgray", edgecolor="black"), 
                                     medianprops=dict(color="black"), whiskerprops=dict(color="black"), 
                                     capprops=dict(color="black"), )

                if show_points:
                    for i, vals in enumerate(data):
                        jitter = (np.random.rand(len(vals)) - 0.5) * 0.15
                        ax_pcc_m.scatter(np.full(len(vals), xpos[i]) + jitter, vals, s=12, c="black", zorder=3)

                ax_pcc_m.set_xticks(xpos)
                ax_pcc_m.set_xticklabels(labels, rotation=20, ha="right")
                ax_pcc_m.set_ylabel(f"PCC – {target}")
                #ax_pcc_m.set_ylim(-1.0, 1.0)
                ax_pcc_m.set_title(f"PCC vs other channels — {condition}")
                fig_pcc_m.tight_layout()
                canvas_pcc_m.draw_idle()

            # ---- Force a default PCC plot on load (pick a pair that actually has data)
            try:
                init_cond, init_tgt = None, None

                # find the first (condition, target) where at least one ref channel has PCC values
                for cond in cond_choices:
                    for tgt in _all_channels_for_pcc:
                        has_data = False
                        for ref in _all_channels_for_pcc:
                            if ref == tgt:
                                continue
                            vals = _get_pcc_values(ref, tgt, cond)
                            if vals is not None and np.size(vals) > 0:
                                has_data = True
                                break
                        if has_data:
                            init_cond, init_tgt = cond, tgt
                            break
                    if init_cond:
                        break

                # fallback if no columns found: use the first entries (plot will show a helpful title)
                if init_cond is None:
                    init_cond = cond_choices[0] if cond_choices else ""
                    init_tgt  = _all_channels_for_pcc[0] if _all_channels_for_pcc else ""

                # set the controls and force one draw (auto_call may not trigger programmatically)
                if hasattr(pcc_controls, "condition") and init_cond:
                    pcc_controls.condition.value = init_cond
                if hasattr(pcc_controls, "target") and init_tgt:
                    pcc_controls.target.value = init_tgt

                pcc_controls()  # guarantees the first render
                canvas_pcc_m.draw_idle()

            except Exception as e:
                viewer.status = f"⚠️ PCC default init failed: {e}"

            # Redraw when filtered points change
            def _on_points_changed_pcc(event=None):
                try:
                    cur_tgt = pcc_controls.target.value if hasattr(pcc_controls, "target") else (_all_channels_for_pcc[0] if _all_channels_for_pcc else "")
                    pcc_controls(target=cur_tgt)  # triggers auto_call
                except Exception:
                    pass

            points.events.data.connect(_on_points_changed_pcc)

        else:
            _remove_dock("PCC (metrics)")

    # ---------- auto-save filtered points to CSV (always) ---------

    _export_path = Path(output_folder) / "filtered_points_export.csv"
    _export_path.parent.mkdir(parents=True, exist_ok=True)

    def _save_filtered_points_to_csv():
        try:
            filtered_coords = points.data
            filtered_props  = points.properties

            # Build dataframe even if no points (write header + empty file)
            data = {
                "frame_index": filtered_coords[:, 0] if filtered_coords.size else np.array([], dtype=float),
                "y": filtered_coords[:, 1] if filtered_coords.size else np.array([], dtype=float),
                "x": filtered_coords[:, 2] if filtered_coords.size else np.array([], dtype=float),
            }
            for prop_name, prop_vals in filtered_props.items():
                data[prop_name] = prop_vals

            # Names/tags come from current coords’ frame index
            if filtered_coords.size:
                data["frame_name"] = [names_valid[int(i)] for i in filtered_coords[:, 0]]
                data["frame_tag"]  = [tags_valid[int(i)]  for i in filtered_coords[:, 0]]
            else:
                data["frame_name"] = []
                data["frame_tag"]  = []

            df_out = pd.DataFrame(data)

            # Explicit column order: tag, name, then everything else
            first_cols = ["frame_tag", "frame_name"]
            ordered_cols = first_cols + [c for c in df_out.columns if c not in first_cols]
            df_out = df_out.reindex(columns=ordered_cols, fill_value=np.nan)

            df_out.to_csv(str(_export_path), index=False)
        except Exception as e:
            # keep UI resilient; just log to napari status
            viewer.status = f"⚠️ Auto-save CSV failed: {e}"

    # initial save (unfiltered or current filtered state)
    _save_filtered_points_to_csv()

    # re-save whenever filtering changes the points layer
    def _on_points_changed_autosave(event=None):
        _save_filtered_points_to_csv()

    points.events.data.connect(_on_points_changed_autosave)

    # ---------- export UI ----------
    # ---------- Analysis Plots (Tabbed) ----------
    # Build one dock with tabs for: Radial Image, Radial Profiles, Boxplot
    # We only add tabs for widgets that actually exist (based on flags).
    for nm in [
        "Radial Condition Viewer", "Radial Line Controls", "Radial Line Profiles",
        "Intensity Boxplot", "Boxplot Controls", "Analysis Plots"
    ]:
        _remove_dock(nm)

    plots_panel = QWidget()
    plots_layout = QVBoxLayout(plots_panel)
    plots_layout.setContentsMargins(6, 6, 6, 6)

    tabs = QTabWidget(plots_panel)

    # --- Tab 1: Radial Image controls ---
    if radial_controls is not None:
        tab_radimg = QWidget()
        tab_radimg_layout = QVBoxLayout(tab_radimg)
        tab_radimg_layout.setContentsMargins(6, 6, 6, 6)
        tab_radimg_layout.addWidget(radial_controls.native)
        tab_radimg_layout.addWidget(radial_colormap_control.native)
        tab_radimg_layout.addWidget(canvas_radimg)
        tab_radimg_layout.addWidget(slider_z) 
        tabs.addTab(tab_radimg, "Radial Image")

        _keep_refs(fig_radimg, ax_radimg, canvas_radimg, slider_z)

    # --- Tab 2: Radial Profiles (controls + Plotly viewer) ---
        if radial_line_profiles is not None:
            tab_profiles = QWidget()
            tab_profiles_layout = QVBoxLayout(tab_profiles)
            tab_profiles_layout.setContentsMargins(6, 6, 6, 6)
            tab_profiles_layout.addWidget(radial_line_profiles.native)
            tab_profiles_layout.addWidget(panel_lp)
            tabs.addTab(tab_profiles, "Radial Profiles")

            _keep_refs(fig_lp, canvas_lp)


    # --- Tab 4: PCC (metrics) ---
    if (pcc_controls is not None) and (panel_pcc_metrics is not None):
        tab_pcc_m = QWidget()
        tab_pcc_m_layout = QVBoxLayout(tab_pcc_m)
        tab_pcc_m_layout.setContentsMargins(6, 6, 6, 6)
        tab_pcc_m_layout.addWidget(pcc_controls.native)
        tab_pcc_m_layout.addWidget(panel_pcc_metrics)
        tabs.addTab(tab_pcc_m, "PCC (metrics)")

    # --- Tab 3: Boxplot (controls + canvas) ---
    if (boxplot_controls is not None) and (panel is not None):
        tab_box = QWidget()
        tab_box_layout = QVBoxLayout(tab_box)
        tab_box_layout.setContentsMargins(6, 6, 6, 6)
        tab_box_layout.addWidget(boxplot_controls.native) 
        tab_box_layout.addWidget(panel)                 
        tabs.addTab(tab_box, "MFI")

    # Only add the dock if at least one tab exists
    if tabs.count() > 0:
        plots_layout.addWidget(tabs)
        viewer.window.add_dock_widget(plots_panel, area="right", name="Analysis Plots")

        try:
            # Use the public mapping of dock widgets
            docks = getattr(viewer.window, "dock_widgets", {})
            dock_widget = docks.get("Analysis Plots", None)

            if dock_widget is not None:
                # These generally work on the dock widget itself
                dock_widget.setMinimumWidth(600)
                dock_widget.setMinimumHeight(700)
                dock_widget.resize(600, 700)
        except Exception as e:
            viewer.status = f"⚠️ Could not resize Analysis Plots dock: {e}"

        _keep_refs(
            plots_panel, tabs,
            radial_controls, radial_line_profiles, panel_lp,
            boxplot_controls, panel,
            pcc_controls, panel_pcc_metrics
        )
