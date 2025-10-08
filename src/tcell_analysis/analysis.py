import gc
import os
# from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import math
import tifffile as tiff

import cv2
import numpy as np
import pandas as pd
import torch
from cellpose import utils
from tifffile import imread, imwrite
from tqdm import tqdm
from skimage.transform import resize

from .extract_cells import extract_cells, save_cells
from .masking.segment import segment_channel
from .metrics import calculate_intensity_metrics, extract_features_for_labels
from .my_utils import convert_nd2_to_tiff, read_any_to_cyx
from .preprocessing.background import bg_remove_rolling_ball, bg_remove_gaussian, bg_remove_tophat
from .visualization.plots import save_all_channels_plot, save_channel_overlays
from typing import Union
from typing import Any, Dict, List

import tempfile
import time
from contextlib import contextmanager

def radial_average_ring(img: np.ndarray) -> np.ndarray:
    img32 = img.astype(np.float64, copy=False)
    H, W = img32.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = np.indices((H, W), dtype=np.float32)
    r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    r_int = r.astype(np.int32)
    r_max = int(r_int.max())
    sums = np.bincount(r_int.ravel(), weights=img32.ravel(), minlength=r_max + 1)
    cnts = np.bincount(r_int.ravel(), minlength=r_max + 1)
    profile = sums / np.maximum(cnts, 1)
    return profile[r_int].astype(np.float32, copy=False)

def ensure_square(img: np.ndarray, auto_square: bool) -> np.ndarray:
    h, w = img.shape
    if h == w:
        return img
    if not auto_square:
        raise ValueError(f"Image is not square (H={h}, W={w}). Use auto_square=True to crop.")
    side = min(h, w)
    y0 = (h - side)//2
    x0 = (w - side)//2
    return img[y0:y0+side, x0:x0+side]

def pad_copy_center(img: np.ndarray, out_shape):
    H, W = out_shape
    h, w = img.shape
    canvas = np.zeros((H, W), dtype=img.dtype)
    y0 = (H - h)//2
    x0 = (W - w)//2
    canvas[y0:y0+h, x0:x0+w] = img
    return canvas

def scale_to(img: np.ndarray, out_shape):
    H, W = out_shape
    return resize(img, (H, W), order=1, preserve_range=True).astype(img.dtype)

def montage_from_stack(stack: np.ndarray, ncols=None, nrows=None):
    N, H, W = stack.shape
    if ncols is None or nrows is None:
        root = math.sqrt(N)
        ncols = int(round(root))
        nrows = int(math.floor(root))
        while ncols * nrows < N:
            nrows += 1
    canvas = np.zeros((nrows*H, ncols*W), dtype=stack.dtype)
    idx = 0
    for r in range(nrows):
        for c in range(ncols):
            if idx >= N: break
            y0, x0 = r*H, c*W
            canvas[y0:y0+H, x0:x0+W] = stack[idx]
            idx += 1
    return canvas

@contextmanager
def timeit(msg):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        print(f"[TIMER] {msg}: {dt:.1f} ms")

@contextmanager
def timeit_cpu(msg: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = (time.perf_counter() - t0) * 1000
        print(f"[TIMER] {msg}: {dt:.1f} ms")

@contextmanager
def timeit_gpu(msg: str):
    if torch.cuda.is_available():
        torch.cuda.synchronize()   # make sure GPU is idle BEFORE timing
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()   # include only the GPU work in this block
        dt = (time.perf_counter() - t0) * 1000
        print(f"[TIMER] {msg}: {dt:.1f} ms")

CODE_VERSION = "2025-09-17.1"


def _to_jsonable(v: Any) -> Any:
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    # torch tensors, if any sneak in:
    try:
        import torch
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().tolist()
    except Exception:
        pass
    return v

def _jsonify_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{k: _to_jsonable(v) for k, v in rec.items()} for rec in records]

def _save_cached_metrics(case_dir: Union[str, Path], results: List[Dict[str, Any]]) -> None:
    p = _cache_metrics_path(case_dir)
    tmp = str(p) + ".tmp"
    data = _jsonify_records(results)
    Path(tmp).write_text(json.dumps(data, indent=2))
    os.replace(tmp, str(p))

def _atomic_replace(tmp_path: str, final_path: str) -> None:
    os.replace(tmp_path, final_path)  # atomic on POSIX/NT

def _atomic_write_png(img: np.ndarray, final_path: str) -> None:
    final = Path(final_path)
    final.parent.mkdir(parents=True, exist_ok=True)
    # Use a temp file **with .png suffix**
    with tempfile.NamedTemporaryFile(dir=final.parent, prefix=final.stem + ".", suffix=".png", delete=False) as tmp:
        tmp_name = tmp.name
    ok = cv2.imwrite(tmp_name, img)
    if not ok:
        # clean up temp file if present
        try: os.remove(tmp_name)
        except Exception: pass
        raise RuntimeError(f"Failed to write PNG (OpenCV): {tmp_name}")
    os.replace(tmp_name, final)

def _atomic_write_tiff(arr: np.ndarray, final_path: str) -> None:
    final = Path(final_path)
    final.parent.mkdir(parents=True, exist_ok=True)
    # Use a temp file **with .tif/.tiff suffix**
    suffix = final.suffix if final.suffix.lower() in {".tif", ".tiff"} else ".tiff"
    with tempfile.NamedTemporaryFile(dir=final.parent, prefix=final.stem + ".", suffix=suffix, delete=False) as tmp:
        tmp_name = tmp.name
    imwrite(tmp_name, arr)
    os.replace(tmp_name, final)

def _atomic_write_csv(df: pd.DataFrame, final_path: str) -> None:
    final = Path(final_path)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_suffix(final.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, final)

def _done_json_path(case_dir: str | Path) -> Path:
    return Path(case_dir) / "DONE.json"

def _lock_path(case_dir: str | Path) -> Path:
    return Path(case_dir) / ".lock"

def _cache_metrics_path(case_dir: str | Path) -> Path:
    return Path(case_dir) / "channel_metrics.json"

def _acquire_lock(lock_path: Path) -> bool:
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False

def _release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink(missing_ok=True)
    except Exception:
        pass

def _outputs_complete(case_dir: str | Path) -> bool:
    case_dir = Path(case_dir)
    has_mask = any(case_dir.glob("*_mask.tiff"))
    has_csv  = (case_dir / "per_cell_features.csv").exists()
    return has_mask and has_csv

def _read_done_version(case_dir: str | Path) -> str | None:
    p = _done_json_path(case_dir)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        return str(data.get("code_version"))
    except Exception:
        return None
def _norm_filters(d: dict | None) -> dict:
    if not d:
        return {}
    out = {}
    for k, v in d.items():
        try:
            a = float(v[0]); b = float(v[1])
            out[str(k)] = [min(a, b), max(a, b)]
        except Exception:
            pass
    return out

def _filters_equal(a: dict | None, b: dict | None, tol: float = 1e-9) -> bool:
    A = _norm_filters(a); B = _norm_filters(b)
    if A.keys() != B.keys():
        return False
    for k in A:
        if abs(A[k][0] - B[k][0]) > tol or abs(A[k][1] - B[k][1]) > tol:
            return False
    return True

def _read_done_filters(case_dir: str | Path) -> dict | None:
    p = _done_json_path(case_dir)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        return data.get("filters")
    except Exception:
        return None

def _purge_case_radials_and_cache(case_dir: str | Path) -> None:
    """Remove stale per-ROI radial dirs and cached per-case metrics so we regenerate."""
    case_dir = Path(case_dir)
    try:
        for d in case_dir.iterdir():
            if d.is_dir() and d.name.startswith("RadAv_"):
                # remove the whole channel radial dir
                for p in d.rglob("*"):
                    try: p.unlink()
                    except Exception: pass
                try: d.rmdir()
                except Exception: pass
    except Exception:
        pass
    # cached metrics
    try:
        _cache_metrics_path(case_dir).unlink(missing_ok=True)
    except Exception:
        pass

def _write_done(case_dir: str | Path, *, tag: str, image: str, channels: list[str],
                code_version: str, filters: dict | None = None, seg_channel: str | None = None) -> None:
    meta = {
        "tag": tag,
        "image": image,
        "channels": list(channels),
        "code_version": code_version,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "filters": _norm_filters(filters),          
        "seg_channel": seg_channel or "",           
    }
    tmp = str(_done_json_path(case_dir)) + ".tmp"
    Path(tmp).write_text(json.dumps(meta, indent=2))
    _atomic_replace(tmp, str(_done_json_path(case_dir)))



def _load_cached_metrics(case_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    p = _cache_metrics_path(case_dir)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, list) else []
    except Exception:
        return []

def _approximate_metrics_from_csv(csv_path: Path, tag: str, image_name: str | None = None) -> list[dict]:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return []
    if image_name is None and "image" in df.columns and len(df):
        image_name = str(df["image"].iloc[0])
    cols = [c for c in df.columns if c.endswith("_mean_intensity")]
    out: list[dict] = []
    for col in cols:
        ch = col[: -len("_mean_intensity")]
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        out.append({"channel": ch, "tag": tag, "image": image_name or "", "N": int(len(vals)), "mean_intensity": float(vals.mean())})
    return out

def bg_removal(img, method="rolling_ball", radius=25, max_side=256):
    if method == "rolling_ball":
        return bg_remove_rolling_ball(img, radius=radius, max_side=max_side)
    if method == "gaussian":
        return bg_remove_gaussian(img, radius=radius, max_side=max_side)
    if method == "tophat":
        return bg_remove_tophat(img, radius=radius, max_side=max_side)
    raise ValueError(method)

def _get_2d_channel(crop: np.ndarray, ci: int) -> np.ndarray:
    """
    Return a 2D (H,W) channel plane from a crop that may be (H,W) or (C,H,W).
    Your extract_cells returns either (H,W) for single-channel or (C,H,W) for multi-channel.
    """
    if crop.ndim == 2:
        return crop  # single-channel path
    if crop.ndim == 3:
        if ci < crop.shape[0]:
            return crop[ci]  # (C,H,W)
        raise IndexError(f"Channel index {ci} out of range for crop shape {crop.shape}")
    raise ValueError(f"Unexpected crop shape {crop.shape}")

def process_tiff(
        tiff_image, 
        tiff_path, 
        channel_names, 
        output_dir, 
        num_workers, 
        tag: str = "",
        save_extracted: bool = True,
        make_qc=False,
        seg_channel: str | None = None,
        feature_thresholds: Dict[str, tuple] | None = None,
        ):

    results = []

    stacked_image = tiff_image
    channel_images = {
        channel_names[i]: stacked_image[i] for i in range(len(channel_names))
    }

    with timeit_cpu("background removal (ICAM1)"):
        if "ICAM1" in channel_images and channel_images["ICAM1"] is not None:
            channel_images["ICAM1"] = bg_removal(
                channel_images["ICAM1"],
                method="rolling_ball",    # try: "gaussian" or "tophat" for big wins
                radius=25,
                max_side=256,
            )

    if make_qc:
        all_channels_path = os.path.join(os.path.dirname(tiff_path), "all_channels.png")
        save_all_channels_plot(channel_images, all_channels_path)

    # choose segmentation channel
    if seg_channel and seg_channel in channel_names:
        seg = seg_channel
    else:
        # fallback: prefer "Actin" if present, else last channel
        seg = "Actin" if "Actin" in channel_names else channel_names[-1]
        if seg_channel and seg_channel not in channel_names:
            print(f"[WARN] seg_channel '{seg_channel}' not found in {channel_names}. Using '{seg}'.")

    seg_image = channel_images[seg]
    with timeit("cellpose segmentation"):
        masks, flows, styles, imgs_dn = segment_channel(
            seg_image,
            gpu=torch.cuda.is_available(),
            diameter=100,
            model_type="cyto3",
            scale=0.4, # scale down the image to 25%
        )

    mask_image = masks[0]
    binary_mask = (mask_image > 0).astype(np.uint8)

    # filenames now use the chosen seg channel name
    _atomic_write_png(binary_mask * 255, os.path.join(os.path.dirname(tiff_path), f"{seg}_mask_binary.png"))
    _atomic_write_tiff(mask_image.astype(np.uint16), os.path.join(os.path.dirname(tiff_path), f"{seg}_mask.tiff"))
    
    frame_name = os.path.splitext(os.path.basename(tiff_path))[0]

    with timeit("extract cells"):
        cell_crops, valid_labels = extract_cells(stacked_image, mask_image, tile_size=512)
    
    features_df = extract_features_for_labels(
        mask_image, channel_images, channel_names, valid_labels, num_workers
    )

    # ⬇Guard against None (or unexpected non-DataFrame)
    if features_df is None:
        features_df = pd.DataFrame()

    # (optional) if the function can return an empty ndarray/list, normalize:
    if not isinstance(features_df, pd.DataFrame):
        features_df = pd.DataFrame(features_df)

    # ---------------- APPLY THRESHOLDS (if any) ----------------
    # Keep only rows within all provided ranges, and derive the label subset.
    selected_label_set = set(int(l) for l in valid_labels)  # default = keep all
    if feature_thresholds and len(features_df):
        mask = np.ones(len(features_df), dtype=bool)
        if "circularity" in feature_thresholds and "circularity" in features_df.columns:
            cmin, cmax = feature_thresholds["circularity"]
            mask &= (features_df["circularity"] >= cmin) & (features_df["circularity"] <= cmax)
        if "equivalent_diameter" in feature_thresholds and "equivalent_diameter" in features_df.columns:
            dmin, dmax = feature_thresholds["equivalent_diameter"]
            mask &= (features_df["equivalent_diameter"] >= dmin) & (features_df["equivalent_diameter"] <= dmax)

        # filter rows + compute selected labels
        features_df = features_df.loc[mask].copy()
        if "label" in features_df.columns:
            selected_label_set = set(int(x) for x in features_df["label"].tolist())
        else:
            # if no label column, fallback to keep all
            selected_label_set = set(int(l) for l in valid_labels)

    features_df.insert(0, "tag", tag)
    features_df.insert(1, "image", os.path.basename(tiff_path))

    csv_out = os.path.join(os.path.dirname(tiff_path), "per_cell_features.csv")
    print(f"[INFO] Writing per-cell features to: {csv_out}")
    _atomic_write_csv(features_df, csv_out)

    # Build a mask that contains ONLY the selected labels for downstream metrics
    mask_filtered = mask_image
    try:
        # fast label exclusion if possible
        if selected_label_set != set(int(l) for l in valid_labels):
            mask_filtered = mask_image.copy()
            # Zero out labels not selected
            keep_vec = np.isin(mask_filtered, list(selected_label_set))
            mask_filtered[~keep_vec] = 0
    except Exception:
        # On any failure, safely keep original
        mask_filtered = mask_image

    # Choose which channels to compute radials for
    RADIAL_CHANNELS = list(channel_names)   # <-- ALL channels now (you asked for this)
    AUTO_SQUARE = True
    proc_dir = Path(os.path.dirname(tiff_path))
    cond_dir = Path(output_dir) / f"res_{tag}"
    cond_dir.mkdir(parents=True, exist_ok=True)

    try:
        for d in proc_dir.iterdir():
            if d.is_dir() and d.name.startswith("RadAv_"):
                for p in d.rglob("*"):
                    try:
                        p.unlink()
                    except Exception:
                        pass
                try:
                    d.rmdir()
                except Exception:
                    pass
    except Exception:
        pass

    # Use ONLY selected labels when saving radials/extracted crops
    selected_labels_sorted = sorted(selected_label_set)

    for ch in RADIAL_CHANNELS:
        if ch not in channel_names:
            print(f"[WARN] Channel '{ch}' not in channel_names={channel_names}")
            continue
        ci = channel_names.index(ch)
        rad_dir = proc_dir / f"RadAv_{ch}"
        rad_dir.mkdir(parents=True, exist_ok=True)

        try:
            for lab in selected_labels_sorted:
                crop = cell_crops.get(lab)
                if crop is None:
                    continue
                roi2d = _get_2d_channel(crop, ci)        # (H,W)
                roi2d = ensure_square(roi2d, AUTO_SQUARE)
                rad = radial_average_ring(roi2d).astype(np.float32, copy=False)
                tiff.imwrite(str(rad_dir / f"Cell_{int(lab):05d}_radAv.tif"),
                             rad)
        except Exception as e:
            print(f"[WARN] Radial save failed in {rad_dir}: {e}")

    # Save extracted cells only for selected labels (optional; keeps disk light)
    if save_extracted:
        # make a pruned dict
        to_save = {lab: cell_crops[lab] for lab in selected_labels_sorted if lab in cell_crops}
        save_cells(to_save, output_dir, frame_name)

    # Compute intensity metrics with the FILTERED mask so summaries honor thresholds
    outlines = utils.outlines_list(mask_filtered)
    if make_qc:
        outlined_path = os.path.join(os.path.dirname(tiff_path), "Channels_with_Outlines.png")
        save_channel_overlays(channel_images, outlines, outlined_path)

    for channel_name in channel_names:
        ch_metrics = calculate_intensity_metrics(
            channel_images[channel_name], mask_filtered   # <-- filtered mask
        )
        ch_metrics["channel"] = channel_name
        ch_metrics["tag"] = tag
        ch_metrics["image"] = os.path.basename(tiff_path)
        results.append(ch_metrics)

    #release large arrays BEFORE returning
    try:
        # Drop per-file heavy objects
        del stacked_image, channel_images
        del masks, mask_image, binary_mask
        # If you created cell crops, drop them too
        try:
            del cell_crops, valid_labels
        except NameError:
            pass
    except Exception:
        pass
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results



def process_image_file(
        file_path, 
        channel_names, 
        input_root, 
        output_root, 
        num_workers, 
        tag: str = "",
        save_extracted: bool = True,
        *,
        skip_if_complete: bool = True,
        code_version: str = CODE_VERSION,
        seg_channel: str | None = None,
        channel_rename_map: Dict[str, str] | None = None,
        feature_thresholds: Dict[str, tuple] | None = None,
    ):
    case_name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"[INFO] ➤ Processing case: {case_name}")
    print('here1')
    condition_name = Path(input_root).name  # e.g. "Ctrl" or "Nivo"
    relative_folder = os.path.relpath(os.path.dirname(file_path), start=input_root)  # path under the condition
    output_folder = os.path.join(output_root, condition_name, relative_folder)
    nd2_base = os.path.splitext(os.path.basename(file_path))[0]
    tiff_output_dir = os.path.join(output_folder, nd2_base)
    os.makedirs(tiff_output_dir, exist_ok=True)

    # Check code version, outputs, and FILTERS
    prev_version = _read_done_version(tiff_output_dir)
    prev_filters = _read_done_filters(tiff_output_dir)
    want_filters = _norm_filters(feature_thresholds)

    complete_now = _outputs_complete(tiff_output_dir)
    same_version = (prev_version == code_version)
    same_filters = _filters_equal(prev_filters, want_filters)

    # Fast skip if complete for this code_version
    if skip_if_complete and complete_now and same_version and same_filters:
        cached = _load_cached_metrics(tiff_output_dir)
        if cached:
            return cached
        csv_p = Path(tiff_output_dir) / "per_cell_features.csv"
        approx = _approximate_metrics_from_csv(csv_p, tag, image_name=None)
        return approx

    # If outputs exist but filters differ, purge stale per-ROI radials + cache
    if complete_now and not same_filters:
        _purge_case_radials_and_cache(tiff_output_dir)

    # Acquire lock to avoid concurrent double work
    lock = _lock_path(tiff_output_dir)
    if not _acquire_lock(lock):
        print(f"[INFO] Skipping {case_name}: locked by another process.")
        cached = _load_cached_metrics(tiff_output_dir)
        return cached

    try:
        
        # Load, collapse Z by max projection, save per-channel TIFFs
        print(channel_rename_map)
        print(channel_names)


        cyx, tiff_path, channel_names = read_any_to_cyx(
            file_path,
            tiff_output_dir,
            z_mode="max",
            t_index=0,
            channel_map=channel_rename_map,
            desired_order=channel_names,
        )
        print(f"[DEBUG] wrote to {tiff_path}; channels={channel_names}; cyx={cyx.shape}")

        results = process_tiff(
            cyx,
            tiff_path,
            channel_names,
            output_folder,
            num_workers,
            tag,
            save_extracted=save_extracted,
            make_qc=False,
            seg_channel=seg_channel,
            feature_thresholds=feature_thresholds, 
        )

        # Cache per-case channel metrics + DONE.json
        _save_cached_metrics(tiff_output_dir, results)
        _write_done(tiff_output_dir, tag=tag, image=os.path.basename(tiff_path),
                    channels=channel_names, code_version=code_version,
                    filters=feature_thresholds, seg_channel=seg_channel,
                    )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results
    finally:
        _release_lock(lock)


def process_folder(
        input_root, 
        channel_names, 
        output_root, 
        num_workers, 
        progress_callback=None, 
        tag: str = "",
        save_extracted: bool = True,
        *,
        skip_if_complete: bool = True,
        code_version: str = CODE_VERSION,
        seg_channel: str | None = None,
        channel_rename_map: Dict[str, str] | None = None,
        feature_thresholds: Dict[str, tuple] | None = None,
    ):

    all_results: list[dict] = []
    tasks: list[tuple] = []
    to_run: list[tuple] = []
    IMAGE_EXTS = (".nd2", ".vsi") 
    
    for dirpath, _, filenames in os.walk(input_root):
        image_files = [f for f in filenames if f.lower().endswith(IMAGE_EXTS)]
        for file_name in image_files:
            file_path = os.path.join(dirpath, file_name)
            tasks.append((file_path, channel_names, input_root, output_root, num_workers, tag, save_extracted, seg_channel, channel_rename_map))

    if not tasks:
        print("[WARNING] No files found in the input directory.")
        return []

    for (file_path, chs, in_root, out_root, n_workers, tg, save_ext, segch, cmap) in tasks:
        condition_name = Path(in_root).name
        relative_folder = os.path.relpath(os.path.dirname(file_path), start=in_root)
        tiff_output_dir = os.path.join(out_root, condition_name, relative_folder, os.path.splitext(os.path.basename(file_path))[0])

        # per-file status
        complete = _outputs_complete(tiff_output_dir)
        prev_ver = _read_done_version(tiff_output_dir)
        prev_filters = _read_done_filters(tiff_output_dir)
        want_filters = _norm_filters(feature_thresholds)
        same_ver = (prev_ver == code_version)
        same_filters = _filters_equal(prev_filters, want_filters)

        if skip_if_complete and complete and same_ver and same_filters:
            cached = _load_cached_metrics(tiff_output_dir)
            if cached:
                all_results.extend(cached)
                continue
            # fallback approx
            csv_p = Path(tiff_output_dir) / "per_cell_features.csv"
            approx = _approximate_metrics_from_csv(csv_p, tg, image_name=Path(file_path).with_suffix(".tiff").name)
            all_results.extend(approx)
            continue

        to_run.append((file_path, chs, in_root, out_root, n_workers, tg, save_ext, segch, cmap))

    print(f"[INFO] Found {len(tasks)} frames total; {len(to_run)} to process, {len(tasks)-len(to_run)} already complete.\n")

    if not to_run:
        return all_results  # everything was already complete

    with ThreadPoolExecutor(max_workers=min(num_workers, len(to_run))) as executor:
        futures = {
            executor.submit(
                process_image_file,
                file_path, chs, in_root, out_root, n_workers, tg, save_ext,
                skip_if_complete=skip_if_complete,
                code_version=code_version,
                seg_channel=segch,             
                channel_rename_map=cmap,  
                feature_thresholds=feature_thresholds,         
            ): (file_path, chs)
            for (file_path, chs, in_root, out_root, n_workers, tg, save_ext, segch, cmap) in to_run
        }
        with tqdm(total=len(futures), desc="Processing frames") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    all_results.extend(result)
                except Exception as e:
                    args = futures[future]
                    print(f"[ERROR] A file failed ({args[0]}): {e}")
                finally:
                    pbar.update(1)
                    if progress_callback:
                        done = pbar.n
                        total = pbar.total
                        progress_callback(done, total)

    return all_results

def _aggregate_radials_for_condition(condition_root: Path, tag: str, channel: str, *, scale_images="No", target_size=None):
    cond_dir = condition_root / f"res_{tag}"   # e.g. output/Ctrl/res_Ctrl
    proc_dirs = sorted([p for p in condition_root.rglob("*") if p.is_dir() and p.name.startswith("RadAv_"+channel)])
    # Collect all per-ROI radials across all Process folders under this output root (for this run)
    rad_files = []
    for rd in proc_dirs:
        # ensure this rad_dir belongs to the same tag by checking it’s under the same tree
        # (if your outputs for multiple tags share the same output_root, you can tighten this filter)
        rad_files += sorted([p for p in rd.glob("*.tif") if p.is_file()])

    if not rad_files:
        print(f"[INFO] No radial files for tag={tag}, channel={channel}")
        return

    # Load ALL (memory heavy) or stream if large:
    imgs = [tiff.imread(str(p)).astype(np.float32) for p in rad_files]

    if target_size is None:
        Hmax = max(im.shape[0] for im in imgs)
        Wmax = max(im.shape[1] for im in imgs)
        target = (Hmax, Wmax)
    else:
        target = tuple(target_size)

    normed = []
    if str(scale_images).lower() == "yes":
        for im in imgs:
            normed.append(scale_to(im, target))
    else:
        for im in imgs:
            normed.append(pad_copy_center(im, target))

    stack = np.stack(normed, axis=0).astype(np.float32)

    cond_dir.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(str(cond_dir / f"{channel}_radStack.tif"), stack)
    tiff.imwrite(str(cond_dir / f"{channel}_radMontage.tif"), montage_from_stack(stack).astype(np.float32))
    tiff.imwrite(str(cond_dir / f"{channel}_radTotAv.tif"), stack.mean(axis=0).astype(np.float32))
    print(f"[INFO] Saved in {cond_dir.name}: {channel}_radStack.tif, {channel}_radMontage.tif, {channel}_radTotAv.tif")


def run_analysis(
        input_folder, 
        output_folder, 
        channel_names, 
        num_workers=1, 
        progress_callback=None, 
        tag: str = "",
        save_extracted: bool = True,
        *,
        skip_if_complete: bool = True,
        code_version: str = CODE_VERSION,
        seg_channel: str | None = None,
        channel_rename_map: Dict[str, str] | None = None,
        feature_thresholds: Dict[str, tuple] | None = None, 
    ):

    input_root = str(Path(input_folder))
    output_root = str(Path(output_folder))
    (Path(output_root) / "run.json").write_text(json.dumps({
        "tag": tag,
        "channels": channel_names,
        "num_workers": num_workers,
        "save_extracted": save_extracted,
        "code_version": code_version,
        "seg_channel": seg_channel,
    }, indent=2))

    if not os.path.exists(input_root):
        raise FileNotFoundError(f"Input folder not found: {input_root}")
    os.makedirs(output_root, exist_ok=True)

    print(f"[INFO] Running analysis on: {input_root}")
    print(f"[INFO] Output will be saved to: {output_root}")
    print(f"[INFO] Channels: {channel_names}")
    print(f"[INFO] Using {num_workers} CPU workers...")

    print(f"[INFO] Segmentation channel: {seg_channel}")
    print(f"[INFO] channel_rename_map: {channel_rename_map}")
    

    results = process_folder(
        input_root, 
        channel_names, 
        output_root, 
        num_workers, 
        progress_callback, 
        tag,
        save_extracted=save_extracted,
        skip_if_complete=skip_if_complete,
        code_version=code_version,
        seg_channel=seg_channel,
        channel_rename_map=channel_rename_map,
        feature_thresholds=feature_thresholds, 
    )

    '''
    df = pd.DataFrame(results)
    if not len(df):
        print("[INFO] No results to summarize.")
    else:
        summary_path = os.path.join(output_root, "global_metrics_summary.csv")
        _atomic_write_csv(df, summary_path)
        print(f"[INFO] Summary saved to {summary_path}")
    '''

    # Aggregate radials for the chosen segmentation channel
    #radial_channels = [seg_channel] if seg_channel else ([channel_names[-1]] if channel_names else [])
    condition_root = Path(output_root) / Path(input_folder).name  # output/<condition>
    for ch in (channel_names or []):
        _aggregate_radials_for_condition(condition_root, tag, ch, scale_images="No", target_size=None)

