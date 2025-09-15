import threading
from typing import List, Tuple

import cv2
import numpy as np
import torch
from cellpose import models, denoise

_MODEL = None
_MODEL_CFG = None
_MODEL_LOCK = threading.Lock()


def _get_model(
    gpu: bool,
    model_type: str = "cyto3",
    restore_type: str | None = None,
):
    global _MODEL, _MODEL_CFG
    use_gpu = bool(gpu and torch.cuda.is_available())
    key = (use_gpu, model_type, restore_type)

    if _MODEL is not None and _MODEL_CFG == key:
        return _MODEL

    with _MODEL_LOCK:
        if _MODEL is None or _MODEL_CFG != key:
            if restore_type:  # denoise path (slower)
                _MODEL = denoise.CellposeDenoiseModel(
                    gpu=use_gpu, model_type=model_type, restore_type=restore_type
                )
            else:             # standard path (faster)
                _MODEL = models.Cellpose(gpu=use_gpu, model_type=model_type)
            _MODEL_CFG = key
    return _MODEL


def segment_channel(
    image: np.ndarray,
    gpu: bool = True,
    diameter: int = 200,
    *,
    model_type: str = "cyto3",
    use_denoise: bool = False,
    normalize_input: bool = False,
    scale: float = 1.0,
) -> Tuple[list, list, list, list]:
    """
    Run Cellpose segmentation on a single-channel 2D image.

    - Reuses a cached model per process to avoid repeated init.
    - Uses torch.inference_mode() to skip autograd overhead.
    - Ensures contiguous float32 input for faster kernels.

    Returns: (masks, flows, styles, imgs_dn) as in Cellpose.
    """
    if image.ndim != 2:
        raise ValueError(f"segment_channel expects a 2D single-channel image; got shape {image.shape}")

    orig_shape = image.shape
    if scale != 1.0:
        new_size = (int(orig_shape[1] * scale), int(orig_shape[0] * scale))
        img_scaled = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        diameter = max(1, int(diameter * scale))  # scale diameter
    else:
        img_scaled = image

    model = _get_model(
        gpu=gpu, model_type=model_type,
        restore_type="denoise_cyto3" if use_denoise else None
    )

    img_scaled = np.ascontiguousarray(img_scaled, dtype=np.float32)

    # Inference only (no gradients, less overhead)
    with torch.inference_mode():
        masks, flows, styles, imgs_dn = model.eval(
            [img_scaled],                     # keep as list to match Cellpose API
            diameter=diameter,
            channels=[0, 0],           # 1-channel grayscale
            do_3D=False,
            normalize=True,
        )

    # Upsample masks back
    if scale != 1.0:
        mask_resized = cv2.resize(
            masks[0].astype(np.uint16),
            (orig_shape[1], orig_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        masks = [mask_resized]

    return masks, flows, styles, imgs_dn


def segment_batch(
    images: List[np.ndarray],
    gpu: bool = True,
    diameter: int = 200,
    *,
    model_type: str = "cyto3",
    restore_type: str = "denoise_cyto3",
) -> Tuple[list, list, list, list]:
    """
    Optional: amortize overhead further by segmenting a small batch of 2D images at once.
    Useful when you have many fields-of-view from the same file and want to reuse CUDA state.

    NOTE: Keep batches modest to avoid VRAM spikes.
    """
    model = _get_model(gpu=gpu, model_type=model_type, restore_type=restore_type)

    imgs: List[np.ndarray] = []
    for im in images:
        if im.ndim != 2:
            raise ValueError(f"segment_batch expects 2D images; got {im.shape}")
        x = im.astype(np.float32, copy=False)
        x = np.ascontiguousarray(x)
        imgs.append(x)

    with torch.inference_mode():
        return model.eval(
            imgs,
            diameter=diameter,
            channels=[0, 0],
            do_3D=False,
            normalize=True,
        )
