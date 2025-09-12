import cv2
import numpy as np
from skimage.restoration import rolling_ball


def remove_background_rolling_ball(image, radius):
    """
    Estimate and subtract background using a downsampled rolling ball.

    Args:
        image (np.ndarray): 2D input image
        radius (int or float): rolling ball radius (will be scaled)

    Returns:
        np.ndarray: background-subtracted image
    """
    orig_shape = image.shape
    downsample_size = min(256, image.shape[0] // 2)
    scale = downsample_size / image.shape[0]

    radius_scaled = max(5, int(radius * scale))

    # Downscale image
    small = cv2.resize(
        image, (downsample_size, downsample_size), interpolation=cv2.INTER_AREA
    )

    # Estimate background on small image
    background_small = rolling_ball(small, radius=radius_scaled)

    # Upscale background to original size
    background = cv2.resize(
        background_small,
        (orig_shape[1], orig_shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )

    # Subtract and clip to non-negative values
    return np.clip(image - background, 0, None)
