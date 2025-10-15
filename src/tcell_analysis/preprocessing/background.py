import cv2
import numpy as np
from skimage.restoration import rolling_ball
from typing import Optional

def _pyramid_levels(H: int, W: int, max_side: Optional[int]) -> int:
    """How many pyrDown steps are needed so max(H, W) <= max_side.
       If max_side is None or <= 0, return 0 (no downsampling)."""
    if max_side is None or max_side <= 0:
        return 0
    levels = 0
    curH, curW = H, W
    while max(curH, curW) > max_side:
        curH = (curH + 1) // 2
        curW = (curW + 1) // 2
        levels += 1
    return levels


def bg_remove_rolling_ball(
    image: np.ndarray,
    radius: int | float,
    max_side: Optional[int] = 256,   # None => no downsampling
    min_radius: int = 5,
) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError(f"expected 2D image; got {image.shape}")

    in_dtype = image.dtype
    H, W = image.shape

    levels = _pyramid_levels(H, W, max_side)

    # Build pyramid (or not)
    small = image
    for _ in range(levels):
        small = cv2.pyrDown(small)  # keeps dtype

    # Scale radius to pyramid scale (no change if levels==0)
    scaled_radius = max(min_radius, int(round(float(radius) / (2 ** levels))))

    # Compute background at small scale in float32
    small_f32 = small.astype(np.float32, copy=False)
    bg_small = rolling_ball(small_f32, radius=scaled_radius).astype(np.float32, copy=False)

    # Upsample back to original size
    bg = bg_small
    for _ in range(levels):
        bg = cv2.pyrUp(bg)
    if bg.shape != (H, W):
        bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_LINEAR)

    # Subtract background preserving dtype
    if np.issubdtype(in_dtype, np.integer):
        out = cv2.subtract(image, np.clip(bg, 0, np.iinfo(in_dtype).max).astype(in_dtype, copy=False))
    else:
        out = np.clip(image.astype(np.float32, copy=False) - bg, 0, None).astype(in_dtype, copy=False)

    return out


def bg_remove_gaussian(
    img: np.ndarray,
    radius: int,
    max_side: Optional[int] = 256,    # None => no downsampling
) -> np.ndarray:
    H, W = img.shape
    levels = _pyramid_levels(H, W, max_side)

    small = img
    for _ in range(levels):
        small = cv2.pyrDown(small)

    # sigma ≈ 0.6 * radius at full res → scale with pyramid
    sigma_small = max(1.0, 0.6 * float(radius) / (2 ** levels))
    blur_small = cv2.GaussianBlur(
        small, (0, 0), sigma_small, sigma_small, borderType=cv2.BORDER_REPLICATE
    )

    bg = blur_small
    for _ in range(levels):
        bg = cv2.pyrUp(bg)
    if bg.shape != (H, W):
        bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_LINEAR)

    out = np.clip(img - bg, 0, None).astype(img.dtype, copy=False)
    return out


def bg_remove_tophat(
    img: np.ndarray,
    radius: int,
    max_side: Optional[int] = 256,    # None => no downsampling
) -> np.ndarray:
    H, W = img.shape
    levels = _pyramid_levels(H, W, max_side)

    small = img
    for _ in range(levels):
        small = cv2.pyrDown(small)

    # Scale structuring element with pyramid level (no change if levels==0)
    radius_small = max(3, int(round(float(radius) / (2 ** levels))))
    k = 2 * radius_small + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg_small = cv2.morphologyEx(small, cv2.MORPH_OPEN, kernel, borderType=cv2.BORDER_REPLICATE)

    bg = bg_small
    for _ in range(levels):
        bg = cv2.pyrUp(bg)
    if bg.shape != (H, W):
        bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_LINEAR)

    out = np.clip(img - bg, 0, None).astype(img.dtype, copy=False)
    return out
