import multiprocessing as mp

import numpy as np
import pandas as pd
from skimage.measure import regionprops


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
    region = next(
        (r for r in regionprops(mask_image) if r.label == region_label), None
    )
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
        record["circularity"] = 0

    label_mask = mask_image == region_label
    for channel_name in channel_names:
        image = channel_images[channel_name]
        pixels = image[label_mask]
        record[f"{channel_name}_mean_intensity"] = pixels.mean()
        record[f"{channel_name}_max_intensity"] = pixels.max()
        record[f"{channel_name}_min_intensity"] = pixels.min()

    return record


def extract_features_for_labels(
    mask_image, channel_images, channel_names, valid_labels, num_workers
):
    """
    Extract per-cell features only for the given label IDs.
    Returns a pandas DataFrame of features (one row per cell).
    """

    if len(valid_labels) == 0:
        print("No valid cells found (all touched border?)")
        return pd.DataFrame()

    # Prepare input for multiprocessing
    region_data_list = [
        (label, mask_image, channel_images, channel_names)
        for label in valid_labels
    ]

    if num_workers <= 1:
        # Run serially
        all_records = [
            compute_features_for_region(rd) for rd in region_data_list
        ]
    else:
        with mp.Pool(processes=num_workers) as pool:
            results = pool.map(compute_features_for_region, region_data_list)

        all_records = [r for r in results if r is not None]

    features_df = pd.DataFrame(all_records)
    features_df = features_df.rename(
        columns={"centroid-0": "y", "centroid-1": "x"}
    )
    features_df["x"] = features_df["x"].round(0).astype(int)
    features_df["y"] = features_df["y"].round(0).astype(int)
    float_cols = features_df.select_dtypes(include=["float"]).columns
    features_df[float_cols] = features_df[float_cols].round(2)

    return features_df
