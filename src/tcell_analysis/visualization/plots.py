import cv2
import matplotlib.pyplot as plt
import numpy as np


def save_channel_overlays(channel_images, outlines, out_path):
    overlayed_images = {}
    for channel_name, img in channel_images.items():
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        rgb = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)
        for outline in outlines:
            outline = np.array(outline, dtype=np.int32)
            cv2.polylines(
                rgb, [outline], isClosed=True, color=(0, 255, 255), thickness=2
            )
        overlayed_images[channel_name] = rgb

    fig, axs = plt.subplots(
        1, len(channel_images), figsize=(5 * len(channel_images), 7)
    )
    for i, (name, img) in enumerate(overlayed_images.items()):
        axs[i].imshow(img)
        axs[i].set_title(name)
        axs[i].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_all_channels_plot(channel_images, out_path):

    n = len(channel_images)
    fig, axs = plt.subplots(1, n, figsize=(5 * n, 7))
    for i, (name, img) in enumerate(channel_images.items()):
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        axs[i].imshow(img_norm, cmap="gray")
        axs[i].set_title(name)
        axs[i].axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
