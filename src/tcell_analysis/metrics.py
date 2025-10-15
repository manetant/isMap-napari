import multiprocessing as mp
from itertools import combinations

import numpy as np
import pandas as pd
from skimage.measure import regionprops
# skimage ≥ 0.25 has this; we’ll fall back to numpy if needed
try:
    from skimage.measure import pearson_corr_coeff as _pearson
except Exception:  # very old scikit-image
    _pearson = None


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Return Pearson r; guard degenerate cases and fall back to numpy if needed."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.size < 2 or b.size < 2:
        return np.nan
    if np.allclose(a.std(), 0.0) or np.allclose(b.std(), 0.0):
        return np.nan
    try:
        if _pearson is not None:
            print(f"DEBUG: Using skimage pearson_corr_coeff")
            return _pearson(a, b)
        # fallback
        r = float(np.corrcoef(a, b)[0, 1])
        return r if np.isfinite(r) else np.nan
    except Exception:
        return np.nan


# Function to calculate intensity metrics for a single image and mask
def calculate_intensity_metrics(image, mask):
    """
    Calculate intensity metrics for the given image inside the mask.

    Parameters:
        image (numpy.ndarray): The channel or combined image.
        mask (numpy.ndarray): The binary mask.

    Returns:
        dict: A dictionary containing intensity metrics (mean, sum, etc.).
    """
    masked_pixels = image[mask > 0]

    return {
        "mean_intensity": (
            int(np.mean(masked_pixels)) if masked_pixels.size > 0 else 0
        ),
        "sum_intensity": (
            int(np.sum(masked_pixels)) if masked_pixels.size > 0 else 0
        ),
        "max_intensity": (
            int(np.max(masked_pixels)) if masked_pixels.size > 0 else 0
        ),
        "min_intensity": (
            int(np.min(masked_pixels)) if masked_pixels.size > 0 else 0
        ),
        "area": int(np.sum(mask > 0)),  # Number of pixels in the mask
    }


def compute_features_for_region(args):
    region_label, mask_image, channel_images, channel_names = args

    # Extract region for this label
    region = next((r for r in regionprops(mask_image) if r.label == region_label), None)
    if region is None:
        return None  # label not found

    record = {
        "label": int(region.label),
        "centroid-0": float(region.centroid[0]),
        "centroid-1": float(region.centroid[1]),
        "area": int(region.area),
        "bbox_area": int(region.bbox_area),
        "convex_area": int(region.convex_area),
        "eccentricity": float(region.eccentricity),
        "equivalent_diameter": float(region.equivalent_diameter),
        "extent": float(region.extent),
        "feret_diameter_max": float(region.feret_diameter_max),
        "major_axis_length": float(region.major_axis_length),
        "minor_axis_length": float(region.minor_axis_length),
        "orientation": float(region.orientation),
        "perimeter": float(region.perimeter),
        "solidity": float(region.solidity),
    }

    if region.perimeter > 0:
        record["circularity"] = 4 * np.pi * region.area / (region.perimeter**2)
    else:
        record["circularity"] = 0.0

    # Mask for this object
    label_mask = (mask_image == region_label)

    # Per-channel intensity summaries inside the object
    for channel_name in channel_names:
        img = channel_images[channel_name]
        pix = img[label_mask]
        # keep raw float for precision here; we’ll round at the end
        record[f"{channel_name}_mean_intensity"] = float(pix.mean()) if pix.size else 0.0
        record[f"{channel_name}_max_intensity"] = float(pix.max()) if pix.size else 0.0
        record[f"{channel_name}_min_intensity"] = float(pix.min()) if pix.size else 0.0

    # Colocalization (PCC) between all channel pairs inside the object
    for chA, chB in combinations(channel_names, 2):
        a = channel_images[chA][label_mask]
        b = channel_images[chB][label_mask]
        r, p = _safe_pearson(a, b)
        # Stable, readable key
        record[f"PCC_{chA}_vs_{chB}"] = float(r)

    return record


def extract_features_for_labels(
    mask_image, channel_images, channel_names, valid_labels, num_workers
):
    """
    Extract per-cell features only for the given label IDs.
    Returns a pandas DataFrame of features (one row per cell),
    including per-channel intensities and pairwise PCC colocalization.
    """

    if len(valid_labels) == 0:
        print("No valid cells found (all touched border?)")
        return pd.DataFrame()

    # Prepare input for multiprocessing
    region_data_list = [
        (label, mask_image, channel_images, channel_names) for label in valid_labels
    ]

    if num_workers <= 1:
        # Run serially
        all_records = [compute_features_for_region(rd) for rd in region_data_list]
    else:
        with mp.Pool(processes=num_workers) as pool:
            results = pool.map(compute_features_for_region, region_data_list)
        all_records = [r for r in results if r is not None]

    features_df = pd.DataFrame(all_records)
    if features_df.empty:
        return features_df

    # Tidy up columns
    features_df = features_df.rename(columns={"centroid-0": "y", "centroid-1": "x"})
    features_df["x"] = features_df["x"].round(0).astype(int)
    features_df["y"] = features_df["y"].round(0).astype(int)

    # Round floats (keep PCC reasonably precise)
    float_cols = features_df.select_dtypes(include=["float"]).columns
    # PCC columns: leave 3 decimals; others keep 2
    pcc_cols = [c for c in float_cols if c.startswith("PCC_")]
    non_pcc_cols = [c for c in float_cols if c not in pcc_cols]

    if non_pcc_cols:
        features_df[non_pcc_cols] = features_df[non_pcc_cols].round(2)
    if pcc_cols:
        features_df[pcc_cols] = features_df[pcc_cols].round(3)

    return features_df
