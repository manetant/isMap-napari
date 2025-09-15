import cv2
import numpy as np
from skimage.restoration import rolling_ball


def bg_remove_rolling_ball(
    image: np.ndarray,
    radius: int | float,
    max_side: int = 256,        # target max size for downsample
    min_radius: int = 5,        # clamp tiny kernels at low scales
) -> np.ndarray:

    if image.ndim != 2:
        raise ValueError(f"expected 2D image; got {image.shape}")

    # Remember input dtype; work in float32 internally only when needed
    in_dtype = image.dtype

    H, W = image.shape
    # Decide how many pyrDown levels we need so max(H, W) <= max_side
    levels = 0
    curH, curW = H, W
    while max(curH, curW) > max_side:
        curH = (curH + 1) // 2
        curW = (curW + 1) // 2
        levels += 1

    # Short-circuit: if already small, skip pyramid
    small = image
    if levels > 0:
        small = image
        for _ in range(levels):
            # pyrDown is faster/more cache-friendly than generic resize
            small = cv2.pyrDown(small)  # preserves dtype

    # Scale radius for the smaller image (divide by 2**levels), clamp
    scaled_radius = max(min_radius, int(round(float(radius) / (2 ** levels))))

    # Run rolling_ball at low res; do in float32 to avoid uint16 overflow
    small_f32 = small.astype(np.float32, copy=False)
    bg_small = rolling_ball(small_f32, radius=scaled_radius).astype(np.float32, copy=False)

    # Upsample background back to original resolution via pyrUp
    bg = bg_small
    for _ in range(levels):
        bg = cv2.pyrUp(bg)  # stays float32

    # Match exact original size (pyrUp may overshoot by 1 px for odd sizes)
    if bg.shape != image.shape:
        bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_LINEAR)

    # Subtract background with dtype preserved (avoid negatives)
    if np.issubdtype(in_dtype, np.integer):
        # Clip and cast in one go
        out = cv2.subtract(image, np.clip(bg, 0, np.iinfo(in_dtype).max).astype(in_dtype, copy=False))
    else:
        out = np.clip(image.astype(np.float32, copy=False) - bg, 0, None)
        out = out.astype(in_dtype, copy=False)

    return out


def bg_remove_gaussian(img: np.ndarray, radius: int, max_side: int = 256) -> np.ndarray:
    H, W = img.shape
    # compute pyramid levels identical to RB
    levels, curH, curW = 0, H, W
    while max(curH, curW) > max_side:
        curH = (curH + 1)//2; curW = (curW + 1)//2; levels += 1

    small = img
    for _ in range(levels):
        small = cv2.pyrDown(small)

    # heuristic: sigma ≈ 0.6 * radius, scaled down by 2**levels
    sigma_small = max(1.0, 0.6 * radius / (2**levels))
    blur_small = cv2.GaussianBlur(small, (0,0), sigma_small, sigma_small,
                                  borderType=cv2.BORDER_REPLICATE)
    bg = blur_small
    for _ in range(levels):
        bg = cv2.pyrUp(bg)

    if bg.shape != img.shape:
        bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_LINEAR)

    out = np.clip(img - bg, 0, None)
    return out.astype(img.dtype, copy=False)


def bg_remove_tophat(img: np.ndarray, radius: int, max_side: int = 256) -> np.ndarray:
    H, W = img.shape
    levels, curH, curW = 0, H, W
    while max(curH, curW) > max_side:
        curH = (curH + 1)//2; curW = (curW + 1)//2; levels += 1

    small = img
    for _ in range(levels):
        small = cv2.pyrDown(small)

    # scale structuring element for small image
    radius_small = max(3, int(round(radius / (2**levels))))
    k = 2*radius_small + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg_small = cv2.morphologyEx(small, cv2.MORPH_OPEN, kernel, borderType=cv2.BORDER_REPLICATE)

    bg = bg_small
    for _ in range(levels):
        bg = cv2.pyrUp(bg)
    if bg.shape != img.shape:
        bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_LINEAR)

    out = np.clip(img - bg, 0, None)
    return out.astype(img.dtype, copy=False)


