import os
import numpy as np
from skimage.measure import regionprops
from tifffile import imwrite


def extract_cells(
    image_or_stack,
    mask_image,
    tile_size: int | None = None,
    exclude_border_touching: bool = True,
):
    """
    Extract each segmented object as cropped arrays based on the mask bounding box.

    Parameters
    ----------
    image_or_stack : np.ndarray
        Either (H, W) single-channel or (C, H, W) multi-channel image.
    mask_image : np.ndarray
        2D labels image where each cell has a unique integer label (>0).
    tile_size : int | None, optional
        If provided, each per-cell crop is center-cropped/padded to (tile_size x tile_size).
        If None (default), crops are returned at their natural bounding-box size.
    exclude_border_touching : bool
        If True, skip objects that touch the image border.

    Returns
    -------
    cell_crops : dict[int, np.ndarray]
        Mapping label_id -> crop. Shape is (H, W) for single-channel, or (C, H, W) for multi-channel.
    valid_labels : list[int]
        List of labels included in the output (after exclusions).
    """
    if image_or_stack.ndim == 2:
        num_channels = 1
    elif image_or_stack.ndim == 3:
        num_channels = image_or_stack.shape[0]
    else:
        raise ValueError(
            f"image_or_stack must be 2D or 3D. Got shape {image_or_stack.shape}"
        )

    image_height, image_width = mask_image.shape
    regions = regionprops(mask_image)

    cell_crops: dict[int, np.ndarray] = {}
    valid_labels: list[int] = []

    for region in regions:
        label_id = region.label
        minr, minc, maxr, maxc = region.bbox

        if exclude_border_touching and (
            minr == 0 or minc == 0 or maxr == image_height or maxc == image_width
        ):
            continue

        valid_labels.append(label_id)

        # Single object mask (H, W)
        single_mask = (mask_image == label_id)

        if num_channels == 1:
            # (H, W)
            object_crop = image_or_stack[minr:maxr, minc:maxc]
            mask_crop = single_mask[minr:maxr, minc:maxc]
            masked_crop = (object_crop * mask_crop).astype(np.float32, copy=False)

            cropped = (
                pad_or_crop(masked_crop, tile_size) if tile_size is not None else masked_crop
            )
        else:
            # (C, H, W)
            object_crop = image_or_stack[:, minr:maxr, minc:maxc]
            mask_crop = single_mask[minr:maxr, minc:maxc][None, :, :]  # (1, h, w) for broadcasting
            masked_crop = (object_crop * mask_crop).astype(np.float32, copy=False)

            if tile_size is not None:
                # pad/crop each channel to tile_size and stack back
                cropped = np.stack(
                    [pad_or_crop(masked_crop[c], tile_size) for c in range(num_channels)],
                    axis=0,
                )
            else:
                cropped = masked_crop

        cell_crops[label_id] = cropped

    return cell_crops, valid_labels


def pad_or_crop(img: np.ndarray, tile_size: int) -> np.ndarray:
    """
    Pads or center-crops a single 2D image to tile_size x tile_size.
    """
    h, w = img.shape

    # Center-crop if too big
    if h > tile_size or w > tile_size:
        center_y, center_x = h // 2, w // 2
        half_tile = tile_size // 2
        start_y = max(center_y - half_tile, 0)
        start_x = max(center_x - half_tile, 0)
        end_y = start_y + tile_size
        end_x = start_x + tile_size
        return img[start_y:end_y, start_x:end_x]

    # Pad if too small (centered)
    pad_h = max(0, (tile_size - h) // 2)
    pad_w = max(0, (tile_size - w) // 2)
    padded = np.pad(
        img,
        ((pad_h, tile_size - h - pad_h), (pad_w, tile_size - w - pad_w)),
        mode="constant",
    )

    return padded[:tile_size, :tile_size]


def save_cells(cell_crops, output_base_dir, frame_name):
    """
    Save extracted cell crops as TIFFs.
    """
    crop_output_dir = os.path.join(output_base_dir, frame_name, "extracted_cells")
    os.makedirs(crop_output_dir, exist_ok=True)

    for label_id, arr in cell_crops.items():
        out_path = os.path.join(crop_output_dir, f"cell_{label_id:04d}.tiff")
        imwrite(out_path, arr)
