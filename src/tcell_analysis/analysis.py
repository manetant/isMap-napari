import gc
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
import torch
from cellpose import utils
from tifffile import imread, imwrite
from tqdm import tqdm

from .extract_cells import extract_cells, save_cells
from .masking.segment import segment_channel
from .metrics import calculate_intensity_metrics, extract_features_for_labels
from .my_utils import convert_nd2_to_tiff, read_any_to_cyx
from .preprocessing.background import bg_remove_rolling_ball, bg_remove_gaussian, bg_remove_tophat
from .visualization.flows import save_flow_quiver_plot
from .visualization.plots import save_all_channels_plot, save_channel_overlays
from typing import Union
from typing import Any, Dict, List

import tempfile
import time
from contextlib import contextmanager

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
    return (case_dir / "Actin_mask.tiff").exists() and (case_dir / "per_cell_features.csv").exists()

def _read_done_version(case_dir: str | Path) -> str | None:
    p = _done_json_path(case_dir)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        return str(data.get("code_version"))
    except Exception:
        return None

def _write_done(case_dir: str | Path, *, tag: str, image: str, channels: list[str], code_version: str) -> None:
    meta = {
        "tag": tag,
        "image": image,
        "channels": list(channels),
        "code_version": code_version,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
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


def process_tiff(
        tiff_image, 
        tiff_path, 
        channel_names, 
        output_dir, 
        num_workers, 
        tag: str = "",
        save_extracted: bool = True,
        make_qc=False):

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

    actin_image = channel_images["Actin"]
    with timeit("cellpose segmentation"):
        masks, flows, styles, imgs_dn = segment_channel(
            actin_image,
            gpu=torch.cuda.is_available(),
            diameter=200,
            model_type="cyto3",
            scale=0.25,   # scale down the image to 25%
        )

    ''' No need to save flows for now
    rgb_flow = flows[0][0]
    flow_yx = flows[0][1]
    cellprob = flows[0][2]
    save_flow_quiver_plot(
        rgb_flow,
        flow_yx,
        cellprob,
        os.path.join(os.path.dirname(tiff_path), "flows.png"),
    )
    '''

    mask_image = masks[0]
    binary_mask = (masks[0] > 0).astype(np.uint8)
    _atomic_write_png(binary_mask * 255, os.path.join(os.path.dirname(tiff_path), "Actin_mask_binary.png"))
    _atomic_write_tiff(mask_image.astype(np.uint16), os.path.join(os.path.dirname(tiff_path), "Actin_mask.tiff"))

    frame_name = os.path.splitext(os.path.basename(tiff_path))[0]

    with timeit("extract cells"):
        cell_crops, valid_labels = extract_cells(stacked_image, mask_image, tile_size=512)
    
    # To save extracted cells
    if save_extracted: 
        save_cells(cell_crops, output_dir, frame_name)

    features_df = extract_features_for_labels(
        mask_image, channel_images, channel_names, valid_labels, num_workers
    )

    # ⬇Guard against None (or unexpected non-DataFrame)
    if features_df is None:
        features_df = pd.DataFrame()

    # (optional) if the function can return an empty ndarray/list, normalize:
    if not isinstance(features_df, pd.DataFrame):
        features_df = pd.DataFrame(features_df)

    features_df.insert(0, "tag", tag)
    features_df.insert(1, "image", os.path.basename(tiff_path))

    csv_out = os.path.join(os.path.dirname(tiff_path), "per_cell_features.csv")
    print(f"[INFO] Writing per-cell features to: {csv_out}")
    _atomic_write_csv(features_df, csv_out)

    outlines = utils.outlines_list(mask_image)
    if make_qc:
        outlined_path = os.path.join(os.path.dirname(tiff_path), "Channels_with_Outlines.png")
        save_channel_overlays(channel_images, outlines, outlined_path)

    for channel_name in channel_names:
        ch_metrics = calculate_intensity_metrics(
            channel_images[channel_name], mask_image
        )
        ch_metrics["channel"] = channel_name
        ch_metrics["tag"] = tag
        ch_metrics["image"] = os.path.basename(tiff_path) 
        results.append(ch_metrics)

    #release large arrays BEFORE returning
    try:
        # Drop per-file heavy objects
        del stacked_image, channel_images, actin_image
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
    ):
    case_name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"[INFO] ➤ Processing case: {case_name}")

    relative_folder = os.path.relpath(os.path.dirname(file_path), start=input_root)
    output_folder = os.path.join(output_root, relative_folder)
    nd2_base = os.path.splitext(os.path.basename(file_path))[0]
    tiff_output_dir = os.path.join(output_folder, nd2_base)
    os.makedirs(tiff_output_dir, exist_ok=True)

    # Fast skip if complete for this code_version
    if skip_if_complete and _outputs_complete(tiff_output_dir) and _read_done_version(tiff_output_dir) == code_version:
        cached = _load_cached_metrics(tiff_output_dir)
        if cached:
            return cached
        # Fallback approximation (older runs without cache file)
        csv_p = Path(tiff_output_dir) / "per_cell_features.csv"
        approx = _approximate_metrics_from_csv(csv_p, tag, image_name=None)
        return approx

    # Acquire lock to avoid concurrent double work
    lock = _lock_path(tiff_output_dir)
    if not _acquire_lock(lock):
        print(f"[INFO] Skipping {case_name}: locked by another process.")
        cached = _load_cached_metrics(tiff_output_dir)
        return cached

    try:
        # Load, collapse Z by max projection, save per-channel TIFFs
        cyx, tiff_path, channel_names = read_any_to_cyx(
            file_path,
            tiff_output_dir,
            z_mode="max",
            t_index=0,
            channel_map={"SD DAPI": "ICAM1", "SD GFP": "pTyr", "SD RFP": "Actin"},
            desired_order=["ICAM1", "pTyr", "Actin"],
        )

        tiff_image = cyx
        results = process_tiff(
            tiff_image,
            tiff_path,
            channel_names,    
            output_folder,
            num_workers,
            tag,
            save_extracted=save_extracted,
        )

        # Cache per-case channel metrics + DONE.json
        _save_cached_metrics(tiff_output_dir, results)
        _write_done(tiff_output_dir, tag=tag, image=os.path.basename(tiff_path),
                    channels=channel_names, code_version=code_version)

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
    ):

    all_results: list[dict] = []
    tasks: list[tuple] = []
    to_run: list[tuple] = []

    for dirpath, _, filenames in os.walk(input_root):
        nd2_files = [f for f in filenames if f.lower().endswith(".nd2")]
        for file_name in nd2_files:
            file_path = os.path.join(dirpath, file_name)
            tasks.append((file_path, channel_names, input_root, output_root, num_workers, tag, save_extracted))

    if not tasks:
        print("[WARNING] No .nd2 files found in the input directory.")
        return []

    # Partition tasks into (skip/submit)
    for (file_path, chs, in_root, out_root, n_workers, tg, save_ext) in tasks:
        relative_folder = os.path.relpath(os.path.dirname(file_path), start=in_root)
        tiff_output_dir = os.path.join(out_root, relative_folder, os.path.splitext(os.path.basename(file_path))[0])

        if skip_if_complete and _outputs_complete(tiff_output_dir) and _read_done_version(tiff_output_dir) == code_version:
            cached = _load_cached_metrics(tiff_output_dir)
            if cached:
                all_results.extend(cached)
                continue
            # fallback approx if cache missing
            csv_p = Path(tiff_output_dir) / "per_cell_features.csv"
            approx = _approximate_metrics_from_csv(csv_p, tg, image_name=os.path.basename(file_path).replace(".nd2", ".tiff"))
            all_results.extend(approx)
            continue

        to_run.append((file_path, chs, in_root, out_root, n_workers, tg, save_ext))

    print(f"[INFO] Found {len(tasks)} ND2 files total; {len(to_run)} to process, {len(tasks)-len(to_run)} already complete.\n")

    if not to_run:
        return all_results  # everything was already complete

    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm

    with ProcessPoolExecutor(max_workers=min(num_workers, len(to_run))) as executor:
        futures = {
            executor.submit(
                process_image_file,
                file_path, chs, in_root, out_root, n_workers, tg, save_ext,
                skip_if_complete=skip_if_complete,
                code_version=code_version,
            ): (file_path, chs)
            for (file_path, chs, in_root, out_root, n_workers, tg, save_ext) in to_run
        }
        with tqdm(total=len(futures), desc="Processing ND2 files") as pbar:
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


def run_analysis(
        input_folder, 
        output_folder, 
        channel_names, 
        num_workers=4, 
        progress_callback=None, 
        tag: str = "",
        save_extracted: bool = True,
        *,
        skip_if_complete: bool = True,
        code_version: str = CODE_VERSION,
    ):

    input_root = str(Path(input_folder))
    output_root = str(Path(output_folder))
    (Path(output_root) / "run.json").write_text(json.dumps({
        "tag": tag,
        "channels": channel_names,
        "num_workers": num_workers,
        "save_extracted": save_extracted,
        "code_version": code_version,
    }, indent=2))

    if not os.path.exists(input_root):
        raise FileNotFoundError(f"Input folder not found: {input_root}")
    os.makedirs(output_root, exist_ok=True)

    print(f"[INFO] Running analysis on: {input_root}")
    print(f"[INFO] Output will be saved to: {output_root}")
    print(f"[INFO] Channels: {channel_names}")
    print(f"[INFO] Using {num_workers} CPU workers...")

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
    )

    df = pd.DataFrame(results)
    if not len(df):
        print("[INFO] No results to summarize.")
    else:
        summary_path = os.path.join(output_root, "global_metrics_summary.csv")
        _atomic_write_csv(df, summary_path)
        print(f"[INFO] Summary saved to {summary_path}")

