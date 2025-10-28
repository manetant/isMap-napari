import os
from bioio import BioImage
import numpy as np
import tifffile as tiff
from pathlib import Path
from nd2reader import ND2Reader
from typing import Dict, List, Optional, Tuple
import time
import logging
from .metadata_utils import get_pixel_sizes_and_units
from .preprocessing.background import (bg_remove_rolling_ball,
                                       bg_remove_gaussian,
                                       bg_remove_tophat)

logger = logging.getLogger(__name__)

# small helper to resolve seg channel -> index after renaming/reordering
def _resolve_seg_index(final_names: list[str], seg_channel: Optional[str | int]) -> Optional[int]:
    if seg_channel is None:
        return None
    if isinstance(seg_channel, int):
        return int(seg_channel) if 0 <= int(seg_channel) < len(final_names) else None
    # name lookup (case/space tolerant like your _norm_name)
    want = _norm_name(seg_channel)
    for i, nm in enumerate(final_names):
        if _norm_name(nm) == want:
            return i
    return None


def _norm_name(s: str) -> str:
    # normalize for matching: lowercase, strip spaces, keep alnum/+-_.
    if s is None:
        return ""
    s = str(s).strip().lower()
    return "".join(ch for ch in s if ch.isalnum() or ch in "+-_.")


def _ome_channel_names_from_tiff(path: str) -> Optional[List[str]]:
    """Parse OME-XML Channel@Name (fallback Channel@Fluor). Return None if absent."""
    # guard: only try for .tif/.tiff
    if not str(path).lower().endswith((".tif", ".tiff")):
        return None
    try:
        with tiff.TiffFile(path) as tf:
            omexml = getattr(tf, "ome_metadata", None)
            if not omexml:
                return None
        try:
            from ome_types import from_xml
            ome = from_xml(omexml)
            for img in ome.images:
                chs = [ch.name or ch.fluor for ch in img.pixels.channels]
                if chs:
                    return chs
        except Exception:
            import xml.etree.ElementTree as ET
            try:
                root = ET.fromstring(omexml)
                chs = []
                for elem in root.iter():
                    if elem.tag.split('}')[-1] == "Channel":
                        name = elem.attrib.get("Name") or elem.attrib.get("Fluor")
                        if name:
                            chs.append(name)
                return chs or None
            except Exception:
                return None
    except Exception:
        return None

def _bg_remove(img: np.ndarray, method: str, radius: int,
               max_side: Optional[int] = None,
               mode: str = "bright") -> np.ndarray:
    """
    mode: "bright" (fluorescence) or "dark" (BF/SIRC)
    """
    method = str(method).lower()
    kwargs = {"radius": radius}
    if max_side is not None:
        kwargs["max_side"] = max_side

    if method == "rolling_ball":
        # Pass mode into bg_remove_rolling_ball if supported
        return bg_remove_rolling_ball(img, mode=mode, **kwargs)
    if method == "gaussian":
        return bg_remove_gaussian(img, **kwargs)
    if method == "tophat":
        return bg_remove_tophat(img, **kwargs)
    raise ValueError(f"Unknown bg method: {method}")

def read_any_to_cyx(
    file_path: str,
    tiff_output_dir: Optional[str] = None,
    z_mode: str = "max",                          # {"max","first","mean"}
    t_index: int = 0,
    channel_map: Optional[Dict[str, str]] = None, # {"SD DAPI":"ICAM1", ...}
    channel_index_map: Optional[Dict[int, str]] = None,
    desired_order: Optional[List[str]] = None,
    *,
    save_ome_tiff: bool = False,                  # write combined OME-TIFF with names
    compression: Optional[str] = "zlib",          # or None/"lzma"/"zstd"
    scene_index: Optional[int] = None,
    apply_bg: bool = False,
    bg_method: str = "rolling_ball",
    bg_radius: int = 50,
    bg_max_side: Optional[int] = None, 
    seg_channel: Optional[str | int] = None, # the channel used for segmentation
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Optional[str]], List[str]]:

    '''
    Returns:
        cyx_proc: np.ndarray (C, Y, X)  - processed (BG removed per settings)
        cyx_raw:  np.ndarray (C, Y, X)  - raw after Z-collapse/reorder/rename
        out_paths: dict with keys {"raw", "processed"} -> saved TIFF paths (or None)
        final_names: list[str] for channels, in order
        px_meta: {"X": float|None, "Y": float|None, "unit": str}
    '''

    print(f"Reading {file_path} ...")

    # Quiet bioformats logging
    os.environ["SCIJAVA_LOG_LEVEL"] = "error"
    os.environ["BIOFORMATS_LOG_LEVEL"] = "ERROR"
    os.environ["JGO_DISABLE_UPDATE"] = "1"

    # ---- open image
    try:
        img = BioImage(file_path)
        # first try BioImage names
        names_src = list(getattr(img, "channel_names", []) or [])
    except Exception as e:
        logger.error(f"Error reading image data from {file_path}: {e}")
        raise

    # sizes / coords for validation + fallback name detection
    try:
        xr = img.xarray_data
        nT = xr.sizes.get("T", 1)
        nS = xr.sizes.get("S", 1)
    except Exception:
        xr = None
        nT = 1
        nS = 1

    if not (0 <= t_index < nT):
        raise IndexError(f"t_index={t_index} out of range for T={nT}")
    if scene_index is not None and not (0 <= scene_index < nS):
        raise IndexError(f"scene_index={scene_index} out of range for S={nS}")

    # ---- get CZYX
    if scene_index is None:
        czyx = img.get_image_data("CZYX", T=t_index)
    else:
        czyx = img.get_image_data("CZYX", T=t_index, S=scene_index)

    # ---- robust channel name detection
    if not names_src or len(names_src) != czyx.shape[0]:
        # try xarray coord labels
        if xr is not None and "C" in xr.coords and xr.coords["C"].size == czyx.shape[0]:
            try:
                vals = xr.coords["C"].values
                names_src = [str(getattr(v, "item", lambda: v)()) if hasattr(v, "item") else str(v) for v in vals]
            except Exception:
                names_src = []
    if (not names_src) or (len(names_src) != czyx.shape[0]):
        # try OME-XML
        ome_names = _ome_channel_names_from_tiff(file_path)
        if ome_names and len(ome_names) == czyx.shape[0]:
            names_src = ome_names
    if (not names_src) or (len(names_src) != czyx.shape[0]):
        names_src = [f"C{i}" for i in range(czyx.shape[0])]

    # ---- Z collapse -> CYX
    if czyx.shape[1] > 1:
        if z_mode == "max":
            cyx = np.nanmax(czyx, axis=1) if np.issubdtype(czyx.dtype, np.floating) else np.max(czyx, axis=1)
        elif z_mode == "first":
            cyx = czyx[:, 0, :, :]
        elif z_mode == "mean":
            m = np.nanmean(czyx, axis=1) if np.issubdtype(czyx.dtype, np.floating) else np.mean(czyx, axis=1)
            if np.issubdtype(czyx.dtype, np.integer):
                m = np.rint(m).astype(czyx.dtype)
            cyx = m
        else:
            raise ValueError("z_mode must be one of {'max','first','mean'}")
    else:
        cyx = np.squeeze(czyx, axis=1)

    # ---- apply rename maps on SOURCE names
    final_names = names_src[:]
    if channel_map:
        cm_norm = {_norm_name(k): v for k, v in channel_map.items()}
        final_names = [cm_norm.get(_norm_name(nm), nm) for nm in final_names]
    if channel_index_map:
        final_names = [channel_index_map.get(i, nm) for i, nm in enumerate(final_names)]

    # ---- STRICT FILTER + ORDER using desired_order (post-rename)
    if desired_order:
        desired_norm = [_norm_name(x) for x in desired_order]
        names_norm = [_norm_name(nm) for nm in final_names]

        idxs = []
        missing = []
        for want_nm, want_norm in zip(desired_order, desired_norm):
            try:
                idxs.append(names_norm.index(want_norm))
            except ValueError:
                missing.append(want_nm)

        if missing:
            logger.warning("desired_order items not found and will be skipped: %s", missing)

        if not idxs:
            raise ValueError(
                f"No requested channels found in file. wanted={desired_order}, have={final_names}"
            )

        cyx = cyx[idxs, ...]
        final_names = [final_names[i] for i in idxs]

    cyx_raw = cyx.astype(np.float32, copy=False)

    # Which index is segmentation channel in this final ordering?
    seg_idx = _resolve_seg_index(final_names, seg_channel)
    print(f"Segmentation channel: {seg_channel}")
    print(f"Final channels: {final_names}, seg_idx={seg_idx}")

    # grab raw seg plane only
    seg_raw = None
    if seg_idx is not None:
        seg_raw = cyx_raw[seg_idx].copy()   # keep just the raw 2D plane


    # Build processed copy
    cyx_proc = cyx_raw.copy()

    bg_meta = {}

    if apply_bg:
        t0 = time.perf_counter()
        for ci, ch_name in enumerate(final_names):
            # default: bright (fluorescence)
            mode = "bright"
            if ch_name=='bf' or ch_name=='sirc':
                mode = "dark"

            cyx_proc[ci] = _bg_remove(
                cyx_proc[ci],
                method=bg_method,
                radius=bg_radius,
                max_side=bg_max_side,
                mode=mode,
            )
        dt = (time.perf_counter() - t0) * 1000.0
        logger.info(f"[BG] Applied {bg_method} (r={bg_radius}) to {cyx_proc.shape[0] - (1 if (seg_idx is not None) else 0)} channels in {dt:.1f} ms")
    else:
        logger.info("[BG] Skipped background removal; leaving cyx_proc == cyx_raw")

    out_paths: Dict[str, Optional[str]] = {"raw": None, "processed": None}
    if tiff_output_dir:
        outdir = Path(tiff_output_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        img_basename = os.path.splitext(os.path.basename(file_path))[0]
        base = outdir / img_basename

        bg_meta = {
            "BackgroundRemoved": bool(apply_bg),
            "BackgroundMethod": (str(bg_method) if apply_bg else None),
            "BackgroundRadius": (int(bg_radius) if apply_bg else None),
            "BackgroundMaxSide": (int(bg_max_side) if (apply_bg and bg_max_side is not None) else None),
            "SegChannel": final_names[seg_idx] if seg_idx is not None else None,
        }
        # pixel sizes
        px_meta_clean = get_pixel_sizes_and_units(img)

        # RAW
        raw_path = str(base.with_suffix(".raw.tiff"))
        tiff.imwrite(
            raw_path,
            cyx_raw,
            ome=bool(save_ome_tiff),
            metadata=(
                {"axes": "CYX", "Channel": {"Name": final_names}, **px_meta_clean}
                if save_ome_tiff else
                {"axes": "CYX", "channel_names": final_names, **bg_meta, **px_meta_clean}
            ),
            bigtiff=True,
            compression=compression,
        )
        out_paths["raw"] = raw_path

        # PROCESSED
        proc_path = str(base.with_suffix(".tiff") if apply_bg else base.with_suffix(".tiff"))
        tiff.imwrite(
            proc_path,
            cyx_proc,
            ome=bool(save_ome_tiff),
            metadata=(
                {"axes": "CYX", "Channel": {"Name": final_names}, **bg_meta, **px_meta_clean}
                if save_ome_tiff else
                {"axes": "CYX", "channel_names": final_names, **bg_meta, **px_meta_clean}
            ),
            bigtiff=True,
            compression=compression,
        )
        out_paths["processed"] = proc_path

    return cyx_proc, seg_raw, out_paths, final_names, px_meta_clean


def convert_nd2_to_tiff(nd2_file_path, channel_names, output_dir):
    """
    Convert all channels of a .nd2 file to a single multi-channel .tiff file.

    Parameters:
        nd2_file_path (str): Path to the .nd2 file.
        channel_names (list): List of channel names in order.
        output_dir (str): Directory to save the .tiff file.

    Returns:
        str: Path to the combined multi-channel TIFF file.
    """
    try:
        with ND2Reader(nd2_file_path) as nd2_file:
            folder_name = os.path.basename(os.path.dirname(nd2_file_path))
            tiff_output_dir = output_dir
            os.makedirs(tiff_output_dir, exist_ok=True)

            # Read each channel into a list
            channel_stack = []
            for i in range(len(channel_names)):
                channel_image = nd2_file.get_frame_2D(c=i)
                channel_stack.append(channel_image)

            # Stack into a 3D array: (C, H, W)
            stacked_image = np.stack(channel_stack, axis=0).astype("float32")

            # Save the multi-channel TIFF
            # Use original filename with .tiff extension
            nd2_basename = os.path.splitext(os.path.basename(nd2_file_path))[0]
            combined_tiff_path = os.path.join(
                tiff_output_dir, f"{nd2_basename}.tiff"
            )
            tiff.imwrite(
                combined_tiff_path,
                stacked_image,
                metadata={"axes": "CHW", "channel_names": channel_names},
            )
            return combined_tiff_path

    except Exception as e:
        print(f"Error processing {nd2_file_path}: {e}")
        return None
