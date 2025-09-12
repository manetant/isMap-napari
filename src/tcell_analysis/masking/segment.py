import threading
from typing import List, Tuple

import numpy as np
import torch
from cellpose import denoise

_MODEL = None
_MODEL_CFG = None
_MODEL_LOCK = threading.Lock()


def _get_model(
    gpu: bool,
    model_type: str = "cyto3",
    restore_type: str = "denoise_cyto3",
):
    """
    Create or return a cached CellposeDenoiseModel with the requested config.
    Caching is per-process. Safe for multi-threaded callers.
    """
    global _MODEL, _MODEL_CFG
    # Verify GPU availability only once per request
    use_gpu = bool(gpu and torch.cuda.is_available())
    key = (use_gpu, model_type, restore_type)

    if _MODEL is not None and _MODEL_CFG == key:
        return _MODEL

    with _MODEL_LOCK:
        if _MODEL is None or _MODEL_CFG != key:
            # Initialize once
            _MODEL = denoise.CellposeDenoiseModel(
                gpu=use_gpu, model_type=model_type, restore_type=restore_type
            )
            _MODEL_CFG = key
    return _MODEL


def segment_channel(
    image: np.ndarray,
    gpu: bool = True,
    diameter: int = 200,
    *,
    model_type: str = "cyto3",
    restore_type: str = "denoise_cyto3",
    normalize_input: bool = False,
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

    model = _get_model(gpu=gpu, model_type=model_type, restore_type=restore_type)

    # Fast, zero-copy-ish preprocessing
    img = np.asarray(image)
    if img.dtype != np.float32:
        img = img.astype(np.float32, copy=False)
    if normalize_input:
        # optional light normalization (kept off to match Cellpose's own normalize=True)
        # img = (img - img.min()) / (img.ptp() + 1e-6)
        pass
    img = np.ascontiguousarray(img)

    # Inference only (no gradients, less overhead)
    with torch.inference_mode():
        masks, flows, styles, imgs_dn = model.eval(
            [img],                     # keep as list to match Cellpose API
            diameter=diameter,
            channels=[0, 0],           # 1-channel grayscale
            do_3D=False,
            normalize=True,            # let Cellpose handle normalization
        )

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
