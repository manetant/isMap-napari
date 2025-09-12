import os
from bioio import BioImage
import numpy as np
import tifffile as tiff
from pathlib import Path
from nd2reader import ND2Reader
from typing import Dict, List, Optional, Tuple

import logging

logger = logging.getLogger(__name__)

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
) -> Tuple[Optional[str], List[str]]:
    """
    Return (combined_tiff_path, final_channel_names).
    combined_tiff_path is the path to the multi-channel TIFF if written, else None.
    """
    img = BioImage(file_path)

    # Validate t_index
    try:
        xr = img.xarray_data  # dims ("T","C","Z","Y","X") usually
        nT = xr.sizes.get("T", 1)
    except Exception:
        xr = None
        nT = 1
    if not (0 <= t_index < nT):
        raise IndexError(f"t_index={t_index} out of range for T={nT}")

    # Load CZYX for single timepoint
    czyx = img.get_image_data("CZYX", T=t_index)  # (C, Z, Y, X)

    # Detect channel labels
    detected: Optional[List[str]] = None
    if xr is not None:
        try:
            if "C" in xr.coords and xr.coords["C"].size == xr.sizes["C"]:
                vals = xr.coords["C"].values
                detected = [str(getattr(v, "item", lambda: v)()) if hasattr(v, "item") else str(v) for v in vals]
        except Exception:
            detected = None
    if not detected:
        detected = _ome_channel_names_from_tiff(file_path)

    if not detected or len(detected) != czyx.shape[0]:
        detected = [f"C{i}" for i in range(czyx.shape[0])]

    # Apply renaming rules
    final_names = detected[:]
    if channel_map:
        name_map_norm = {_norm_name(k): v for k, v in channel_map.items()}
        final_names = [name_map_norm.get(_norm_name(nm), nm) for nm in final_names]
    if channel_index_map:
        final_names = [channel_index_map.get(i, nm) for i, nm in enumerate(final_names)]

    # Z collapse -> CYX
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

    # Reorder (soft)
    if desired_order:
        norm_to_idx = {_norm_name(nm): i for i, nm in enumerate(final_names)}
        desired_norm = [_norm_name(x) for x in desired_order]
        first_idxs = [norm_to_idx[nm] for nm in desired_norm if nm in norm_to_idx]
        missing = [desired_order[i] for i, nm in enumerate(desired_norm) if nm not in norm_to_idx]
        if missing:
            logger.warning("desired_order items not found and ignored: %s", missing)
        keep = (first_idxs + [i for i in range(len(final_names)) if i not in first_idxs]) if first_idxs else list(range(len(final_names)))
        cyx = cyx[keep]
        final_names = [final_names[i] for i in keep]
    
    cyx = cyx.astype(np.float32, copy=False)

    combined_tiff_path = None
    if tiff_output_dir:
        outdir = Path(tiff_output_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        # per-channel TIFFs
        seen = set()
        for c_idx, cname in enumerate(final_names):
            safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(cname))[:80]
            while safe in seen:
                safe += "_dup"
            seen.add(safe)
            tiff.imwrite(
                (outdir / f"{safe}.tiff").as_posix(),
                cyx[c_idx],
                photometric="minisblack",
                bigtiff=True,
                compression=compression,
            )

        # combined multi-channel TIFF (optionally OME-TIFF)
        img_basename = os.path.splitext(os.path.basename(file_path))[0]
        combined_tiff_path = os.path.join(tiff_output_dir, f"{img_basename}.tiff")

        if save_ome_tiff:
            # OME axes order "CYX", embed channel names
            tiff.imwrite(
                combined_tiff_path,
                cyx,
                ome=True,
                metadata={"axes": "CYX", "Channel": {"Name": final_names}},
                bigtiff=True,
                compression=compression,
            )
        else:
            tiff.imwrite(
                combined_tiff_path,
                cyx,
                metadata={"axes": "CYX", "channel_names": final_names},
                bigtiff=True,
                compression=compression,
            )

    return cyx, combined_tiff_path, final_names


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
