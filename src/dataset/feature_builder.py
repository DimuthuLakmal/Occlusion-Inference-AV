"""
Pure, stateless feature-building functions shared between offline training
(`OGMDataset`, src/dataset/dataset.py) and the live inference service
(src/serving/inference_service.py). Keeping this logic in one place avoids
train/serve skew: both paths must build tensors identically.
"""
import math
import os
from typing import Tuple

import cv2
import numpy as np

from src.utils.semantic_maps import semantic_map_to_one_hot, SEMANTIC_PALETTE_BGR


def load_background_images(dataset_dir, scene_ids, start_scene=None, end_scene=None):
    """
    Loads `<dataset_dir>/semantic_maps/<scene_id>_background.png` for each scene id
    (optionally filtered to [start_scene, end_scene]) into {int(scene_id): BGR uint8 image}.
    """
    background_images = {}
    for scene_id in scene_ids:
        scene_id_int = int(scene_id)
        if start_scene is not None and scene_id_int < start_scene:
            continue
        if end_scene is not None and scene_id_int > end_scene:
            continue

        bg_path = os.path.join(dataset_dir, 'semantic_maps', f"{scene_id}_background.png")
        img = cv2.imread(bg_path)
        background_images[scene_id_int] = img

    return background_images


def load_semantic_maps(background_images, output_size=(224, 224)):
    """
    Converts each already-loaded background image into the one-hot semantic class map
    [K,H,W] the model expects as `map_obs`, matching OGMDataset.load_maps exactly. Takes
    the same dict `load_background_images` returns, so its key set (scene_id -> image) is
    always in sync with `background_images` by construction - no separate scene-range
    filtering or NOT_FOUND bookkeeping needed downstream.
    """
    semantic_maps = {}
    for scene_id, img in background_images.items():
        one_hot, _, _ = semantic_map_to_one_hot(
            image_bgr=img, palette_bgr=SEMANTIC_PALETTE_BGR, output_size=output_size)
        semantic_maps[scene_id] = one_hot

    return semantic_maps


def compute_last_recorded_t(historical_adjacent_obs):
    """
    historical_adjacent_obs: {track_id: obs[T, F]} (raw, un-normalized).
    Returns {track_id: index of the last timestep that is not all-zero (None if every
    timestep is all-zero, i.e. the vehicle was never actually recorded)}.
    """
    last_recorded_t = {}
    for veh_index, obs in historical_adjacent_obs.items():
        mask = np.any(np.array(obs) != 0, axis=1)
        last_t = np.where(mask)[0].max() if np.any(mask) else None
        last_recorded_t[veh_index] = last_t
    return last_recorded_t


def extract_edge_info(historical_adjacent_obs, hidden_ogm_cells, last_recorded_t) -> Tuple[list, list]:
    """
    Builds a full bipartite graph between adjacent vehicles and hidden cells: one edge per
    (vehicle, cell) pair, weighted by the Euclidean distance between the cell's (x, y) and
    the vehicle's (x, y) at its last recorded timestep. `historical_adjacent_obs` and
    `hidden_ogm_cells` must be in the same (normalized) coordinate space.
    """
    edge_weights = []
    edge_src, edge_dst = [], []
    for i, cell in enumerate(hidden_ogm_cells):
        for j, (track_idx, obs) in enumerate(historical_adjacent_obs.items()):
            # Calculate the distance from the cell to the track center
            obs_last_t = last_recorded_t[track_idx]
            distance = math.sqrt((cell[0] - obs[obs_last_t][0]) ** 2 + (cell[1] - obs[obs_last_t][1]) ** 2)
            edge_weights.append(distance)

            edge_src.append(j)  # Source index is the track index
            edge_dst.append(i)  # Destination index is the cell index

    edge_index = [edge_src, edge_dst]
    return edge_weights, edge_index


def build_vehicle_tensor(historical_adjacent_obs, scene_id):
    """
    historical_adjacent_obs: {track_id: obs[T, 10]} (raw, un-normalized, dict order = vehicle order).
    Returns (historical_adjacent_input[N, T, 10], seq_mask[N, T]) ready for TemporalEncoder:
    class-id fixup (car 0 -> 4), scene_id feature attached + normalized, heading/speed-ish
    columns normalized, and the raw column-9 ("distance to ego") dropped.
    """
    historical_adjacent_obs = np.array(list(historical_adjacent_obs.values()), dtype=np.float32)

    seq_mask = np.all(historical_adjacent_obs == 0, axis=-1)  # all-zero timestep = padding/missing

    # Fixing a class type issue (0 is used to represent car type. Replacing 0 with 4)
    veh_type = historical_adjacent_obs[..., 7]
    mask = (veh_type == 0) & (~seq_mask)
    veh_type[mask] = 4
    historical_adjacent_obs[..., 7] = veh_type

    # Attaching scene_id as a feature
    scene_id_norm = scene_id / 10  # will be divided it further later to bring the range of 0 and 1
    scene_id_arr = np.full(historical_adjacent_obs.shape[:-1] + (1,), scene_id_norm,
                           dtype=historical_adjacent_obs.dtype)
    historical_adjacent_obs = np.concatenate([historical_adjacent_obs, scene_id_arr], axis=-1)

    historical_adjacent_obs[:, :, 2:3] = historical_adjacent_obs[:, :,
                                         2:3] / 360.0  # Normalize heading to [0, 1]. This is a mistake done when extracting the data
    historical_adjacent_obs[:, :, 3:] = historical_adjacent_obs[:, :, 3:] / 10.0

    historical_adjacent_input = np.concatenate((historical_adjacent_obs[:, :, :9],
                                                historical_adjacent_obs[:, :, 10:11]), axis=-1)

    return historical_adjacent_input, seq_mask
