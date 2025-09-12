import gc
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
import torch
from cellpose import utils
from tifffile import imread, imwrite
from tqdm import tqdm

from .extract_cells import extract_cells, save_cells
from .masking.segment import segment_channel
from .metrics import calculate_intensity_metrics, extract_features_for_labels
from .my_utils import convert_nd2_to_tiff, read_any_to_cyx
from .preprocessing.background import remove_background_rolling_ball
from .visualization.flows import save_flow_quiver_plot
from .visualization.plots import save_all_channels_plot, save_channel_overlays


def process_tiff(
        tiff_image, 
        tiff_path, 
        channel_names, 
        output_dir, 
        num_workers, 
        tag: str = "",
        save_extracted: bool = True,
        make_qc=False):

    results = []

    stacked_image = tiff_image
    channel_images = {
        channel_names[i]: stacked_image[i] for i in range(len(channel_names))
    }

    if "ICAM1" in channel_images and channel_images["ICAM1"] is not None:
        channel_images["ICAM1"] = remove_background_rolling_ball(
            channel_images["ICAM1"], radius=25
        )
    if make_qc:
        all_channels_path = os.path.join(os.path.dirname(tiff_path), "all_channels.png")
        save_all_channels_plot(channel_images, all_channels_path)

    actin_image = channel_images["Actin"]
    masks, flows, styles, imgs_dn = segment_channel(
        actin_image, gpu=torch.cuda.is_available(), diameter=200
    )

    ''' No need to save flows for now
    rgb_flow = flows[0][0]
    flow_yx = flows[0][1]
    cellprob = flows[0][2]
    save_flow_quiver_plot(
        rgb_flow,
        flow_yx,
        cellprob,
        os.path.join(os.path.dirname(tiff_path), "flows.png"),
    )
    '''

    mask_image = masks[0]
    binary_mask = (masks[0] > 0).astype(np.uint8)
    cv2.imwrite(
        os.path.join(os.path.dirname(tiff_path), "Actin_mask_binary.png"),
        binary_mask * 255,
    )
    imwrite(
        os.path.join(os.path.dirname(tiff_path), "Actin_mask.tiff"),
        mask_image.astype(np.uint16),
    )

    frame_name = os.path.splitext(os.path.basename(tiff_path))[0]

    cell_crops, valid_labels = extract_cells(stacked_image, mask_image, tile_size=512)
    
    # To save extracted cells
    if save_extracted: 
        save_cells(cell_crops, output_dir, frame_name)

    features_df = extract_features_for_labels(
        mask_image, channel_images, channel_names, valid_labels, num_workers
    )

    # ⬇Guard against None (or unexpected non-DataFrame)
    if features_df is None:
        features_df = pd.DataFrame()

    # (optional) if the function can return an empty ndarray/list, normalize:
    if not isinstance(features_df, pd.DataFrame):
        features_df = pd.DataFrame(features_df)

    features_df.insert(0, "tag", tag)
    features_df.insert(1, "image", os.path.basename(tiff_path))

    csv_out = os.path.join(os.path.dirname(tiff_path), "per_cell_features.csv")
    print(f"[INFO] Writing per-cell features to: {csv_out}")
    features_df.to_csv(csv_out, index=False)

    outlines = utils.outlines_list(mask_image)
    if make_qc:
        outlined_path = os.path.join(os.path.dirname(tiff_path), "Channels_with_Outlines.png")
        save_channel_overlays(channel_images, outlines, outlined_path)

    for channel_name in channel_names:
        ch_metrics = calculate_intensity_metrics(
            channel_images[channel_name], mask_image
        )
        ch_metrics["channel"] = channel_name
        ch_metrics["tag"] = tag
        ch_metrics["image"] = os.path.basename(tiff_path) 
        results.append(ch_metrics)

    return results



def process_image_file(
        file_path, 
        channel_names, 
        input_root, 
        output_root, 
        num_workers, 
        tag: str = "",
        save_extracted: bool = True,):
    
    case_name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"[INFO] ➤ Processing case: {case_name}")

    relative_folder = os.path.relpath(os.path.dirname(file_path), start=input_root)
    output_folder = os.path.join(output_root, relative_folder)
    nd2_base = os.path.splitext(os.path.basename(file_path))[0]
    tiff_output_dir = os.path.join(output_folder, nd2_base)
    os.makedirs(tiff_output_dir, exist_ok=True)


    # Load, collapse Z by max projection, save per-channel TIFFs
    cyx, tiff_path, channel_names = read_any_to_cyx(
        file_path,
        tiff_output_dir,
        z_mode="max",
        t_index=0,
        channel_map={"SD DAPI": "ICAM1", "SD GFP": "pTyr", "SD RFP": "Actin"},
        desired_order=["ICAM1", "pTyr", "Actin"],  # optional, enforces this output order
    )


    #tiff_path = convert_nd2_to_tiff(file_path, channel_names, tiff_output_dir)
    #if tiff_path is None or not os.path.isfile(tiff_path):
    #    raise FileNotFoundError(f"[ERROR] TIFF conversion failed or file not found: {tiff_path}")

    tiff_image = cyx #imread(tiff_path)
    results = process_tiff(
        tiff_image,
        tiff_path,
        channel_names,    
        output_folder,
        num_workers,
        tag,
        save_extracted=save_extracted,
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def process_folder(
        input_root, 
        channel_names, 
        output_root, 
        num_workers, 
        progress_callback=None, 
        tag: str = "",
        save_extracted: bool = True, ):

    all_results = []
    tasks = []

    for dirpath, _, filenames in os.walk(input_root):
        nd2_files = [f for f in filenames if f.lower().endswith(".nd2")]
        for file_name in nd2_files:
            file_path = os.path.join(dirpath, file_name)
            tasks.append((file_path, channel_names, input_root, output_root, num_workers, tag, save_extracted))

    if not tasks:
        print("[WARNING] No .nd2 files found in the input directory.")
        return []

    print(f"[INFO] Found {len(tasks)} ND2 files to process.\n")

    with ProcessPoolExecutor(
        max_workers=min(num_workers, len(tasks))
    ) as executor:
        futures = {
            executor.submit(process_image_file, *args): idx
            for idx, args in enumerate(tasks)
        }
        for i, future in enumerate(
            tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing ND2 files",
            )
        ):
            try:
                result = future.result()
                all_results.extend(result)
            except Exception as e:
                print(f"[ERROR] A file failed: {e}")
            if progress_callback:
                progress_callback(i + 1, len(futures))

    return all_results


def run_analysis(
        input_folder, 
        output_folder, 
        channel_names, 
        num_workers=4, 
        progress_callback=None, 
        tag: str = "",
        save_extracted: bool = True, ):

    input_root = str(Path(input_folder))
    output_root = str(Path(output_folder))
    (Path(output_root) / "run.json").write_text(json.dumps({
        "tag": tag,
        "channels": channel_names,
        "num_workers": num_workers,
        "save_extracted": save_extracted,
    }, indent=2))

    if not os.path.exists(input_root):
        raise FileNotFoundError(f"Input folder not found: {input_root}")
    os.makedirs(output_root, exist_ok=True)

    print(f"[INFO] Running analysis on: {input_root}")
    print(f"[INFO] Output will be saved to: {output_root}")
    print(f"[INFO] Channels: {channel_names}")
    print(f"[INFO] Using {num_workers} CPU workers...")

    results = process_folder(
        input_root, 
        channel_names, 
        output_root, 
        num_workers, 
        progress_callback, 
        tag,
        save_extracted=save_extracted,
        )


    df = pd.DataFrame(results)
    summary_path = os.path.join(output_root, "global_metrics_summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"[INFO] Summary saved to {summary_path}")
