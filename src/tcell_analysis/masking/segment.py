import torch
from cellpose import denoise


def segment_channel(image, gpu=True, diameter=200):
    """
    Run Cellpose segmentation on channel image.

    Parameters:
        image (np.ndarray): 2D image of a single channel
        gpu (bool): Whether to try GPU. Will fallback to CPU if unavailable.
        diameter (int): Expected cell diameter.

    Returns:
        masks: list of instance-labeled masks
        flows: flow outputs (for visualization)
        styles: style vectors
        imgs_dn: denoised images
    """
    # Check if GPU is actually available
    if gpu and not torch.cuda.is_available():
        gpu = False

    # Initialize the denoising model
    model = denoise.CellposeDenoiseModel(
        gpu=gpu, model_type="cyto3", restore_type="denoise_cyto3"
    )

    # Run segmentation
    masks, flows, styles, imgs_dn = model.eval(
        [image], diameter=diameter, channels=[0, 0]
    )

    return masks, flows, styles, imgs_dn
