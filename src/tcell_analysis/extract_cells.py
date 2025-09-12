import os

import numpy as np
from skimage.measure import regionprops
from tifffile import imwrite


def extract_and_save_cells(
    segment_channel,
    mask_image,
    output_base_dir,
    frame_name,
    tile_size=512,
    exclude_border_touching=True,
):
    """
    Extracts each segmented object from the Actin channel (single or multi-channel),
    crops or pads to tile_size x tile_size, and saves as TIFF preserving float32 data.

    Parameters:
        segment_channel (np.ndarray): 2D (H,W) or 3D (C,H,W) float32 image.
        mask_image (np.ndarray): 2D labeled segmentation mask (uint16).
        output_base_dir (str): Root folder for output.
        frame_name (str): Used to create subfolder.
    """
    # Make frame-specific subfolder
    frame_output_dir = os.path.join(output_base_dir, frame_name)
    crop_output_dir = os.path.join(frame_output_dir, "extracted_cells")

    os.makedirs(crop_output_dir, exist_ok=True)

    # Determine channels
    if segment_channel.ndim == 2:
        num_channels = 1
    elif segment_channel.ndim == 3:
        num_channels = segment_channel.shape[0]
    else:
        raise ValueError(
            f"segment_channel must be 2D or 3D. Got shape {segment_channel.shape}"
        )

    image_height, image_width = mask_image.shape
    regions = regionprops(mask_image)
    valid_labels = []

    count = 0

    for region in regions:
        label_id = region.label

        # Exclude border-touching
        minr, minc, maxr, maxc = region.bbox
        if exclude_border_touching and (
            minr == 0
            or minc == 0
            or maxr == image_height
            or maxc == image_width
        ):
            continue

        valid_labels.append(label_id)

        # Binary mask for this object, converted to float32
        single_mask = (mask_image == label_id).astype(segment_channel.dtype)

        # Crop region
        mask_crop = single_mask[minr:maxr, minc:maxc]

        if num_channels == 1:
            object_crop = segment_channel[minr:maxr, minc:maxc]
            masked_crop = object_crop * mask_crop
        else:
            object_crop = segment_channel[:, minr:maxr, minc:maxc]
            masked_crop = object_crop * mask_crop[np.newaxis, :, :]

        # Pad or crop to tile_size
        if num_channels == 1:
            cropped = pad_or_crop(masked_crop, tile_size)
        else:
            cropped = np.stack(
                [
                    pad_or_crop(masked_crop[c], tile_size)
                    for c in range(num_channels)
                ],
                axis=0,
            )

        # Save as float32 TIFF
        out_path = os.path.join(crop_output_dir, f"cell_{label_id:04d}.tiff")
        imwrite(out_path, cropped.astype(np.float32))
        count += 1

    return valid_labels


def pad_or_crop(img, tile_size):
    """
    Pads or crops a single 2D image to tile_size x tile_size.
    """
    h, w = img.shape

    # Crop center if too big
    if h > tile_size or w > tile_size:
        center_y, center_x = h // 2, w // 2
        half_tile = tile_size // 2
        start_y = max(center_y - half_tile, 0)
        start_x = max(center_x - half_tile, 0)
        end_y = start_y + tile_size
        end_x = start_x + tile_size
        return img[start_y:end_y, start_x:end_x]

    # Pad if too small
    pad_h = max(0, (tile_size - h) // 2)
    pad_w = max(0, (tile_size - w) // 2)
    padded = np.pad(
        img,
        ((pad_h, tile_size - h - pad_h), (pad_w, tile_size - w - pad_w)),
        mode="constant",
    )

    return padded[:tile_size, :tile_size]
