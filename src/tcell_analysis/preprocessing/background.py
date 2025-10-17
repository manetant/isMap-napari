import cv2
import numpy as np
from skimage.restoration import rolling_ball
from typing import Optional, Literal, Tuple

Mode = Literal["bright", "dark", "auto"]
Floor = Literal["none", "zero", "percentile", "shift"]

def _pyramid_levels(h: int, w: int, max_side: Optional[int]) -> int:
    if max_side is None:
        return 0
    levels = 0
    while max(h, w) > max_side and min(h, w) >= 2 and levels < 6:
        h, w = (h + 1) // 2, (w + 1) // 2
        levels += 1
    return levels

def _safe_float(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32, copy=False)
    if not np.isfinite(f).all():
        finite = np.isfinite(f)
        fill = np.median(f[finite]) if finite.any() else 0.0
        f[~finite] = fill
    return f

def _auto_polarity(img_f: np.ndarray, bg_f: np.ndarray) -> str:
    r_bright = img_f - bg_f
    r_dark   = bg_f - img_f
    p_b = np.percentile(r_bright, 99) - np.median(r_bright)
    p_d = np.percentile(r_dark,   99) - np.median(r_dark)
    return "bright" if p_b >= p_d else "dark"

def bg_remove_rolling_ball(
    image: np.ndarray,
    radius: int | float,
    max_side: Optional[int] = 256,
    min_radius: int = 3,
    mode: Mode = "bright",
    restore_dtype: bool = False,
    floor_mode: Floor = "zero",           # how to handle negatives
    floor_percentile: float = 1.0,        # used when floor_mode=="percentile"
    return_bg: bool = False,              # optionally return the bg
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """
    Rolling-ball background removal with polarity + optional flooring.
    Returns float32 unless restore_dtype=True.
    If return_bg=True, returns (out, bg).
    """
    if image.ndim != 2:
        raise ValueError(f"expected 2D image; got {image.shape}")

    in_dtype = image.dtype
    H, W = image.shape
    levels = _pyramid_levels(H, W, max_side)

    # Downsample pyramid
    small = image
    for _ in range(levels):
        small = cv2.pyrDown(small)

    # Radius at pyramid scale
    scaled_radius = max(min_radius, int(round(float(radius) / (2 ** levels))))
    scaled_radius = min(scaled_radius, max(1, min(small.shape)//2))

    # Background at small scale
    small_f32 = _safe_float(small)
    bg_small  = rolling_ball(small_f32, radius=scaled_radius).astype(np.float32, copy=False)

    # Upsample bg
    bg = bg_small
    for _ in range(levels):
        bg = cv2.pyrUp(bg)
    if bg.shape != (H, W):
        bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_LINEAR)

    img_f = _safe_float(image)

    # Polarity
    use_mode = _auto_polarity(img_f, bg) if mode == "auto" else mode
    out = img_f - bg if use_mode == "bright" else bg - img_f

    # Flooring options
    if floor_mode == "zero":
        out = np.maximum(out, 0.0)
    elif floor_mode == "percentile":
        # subtract low baseline then floor
        b = np.percentile(out, floor_percentile)
        out = np.maximum(out - b, 0.0)
    elif floor_mode == "shift":
        # shift so min is zero, preserves linearity before integer cast
        mn = np.min(out)
        if np.isfinite(mn) and mn < 0:
            out = out - mn

    # Optional dtype restore (after flooring)
    if restore_dtype:
        if np.issubdtype(in_dtype, np.integer):
            info = np.iinfo(in_dtype)
            out = np.clip(out, info.min, info.max).astype(in_dtype, copy=False)
        else:
            out = out.astype(in_dtype, copy=False)

    if return_bg:
        return out, bg.astype(out.dtype, copy=False)
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
