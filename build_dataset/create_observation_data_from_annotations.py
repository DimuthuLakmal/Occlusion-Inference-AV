import argparse
import json
import math
import os
import pickle
import re
from copy import deepcopy
from pathlib import Path
import random
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import yaml

from src.utils.ogm_util import create_OGM_ego
from src.utils.tracks_import import read_from_csv


def _get_current_index(track_meta, t: int) -> int:
    initial_frame = track_meta["initialFrame"]
    current_index = t - initial_frame
    return current_index


def _get_heading(track, track_meta, t) -> float:
    current_index = _get_current_index(track_meta, t)
    if current_index < 0:
        return None

    return track["heading"][current_index]

def _extract_track_info(track, track_meta, t: int, height_width: tuple, scale_down_factor: float) -> dict:
    current_index = _get_current_index(track_meta, t)
    if current_index < 0:
        return {}

    object_class = track_meta["class"]
    if track["bboxVis"] is not None:
        bounding_box = track["bboxVis"][current_index] / scale_down_factor
    else:
        bounding_box = None
    center_points = track["centerVis"] / scale_down_factor
    center_point = center_points[current_index]

    if bounding_box is not None:
        pts = bounding_box.astype(int)
        pts = pts.reshape((-1, 1, 2))

    else:
        x, y = center_point
        square_coords = [
            (x - 1, y - 1),
            (x + 1, y - 1),
            (x + 1, y + 1),
            (x - 1, y + 1),
            (x - 1, y - 1)
        ]

        # bbox = plt.Polygon(square_coords, closed=True)
        pts = np.array(square_coords, np.int32)
        # pts = bounding_box.astype(int)
        pts = pts.reshape((-1, 1, 2))

    heading = track['heading'][current_index]
    x_velocity = track["xVelocity"][current_index]
    y_velocity = track["yVelocity"][current_index]
    x_acceleration = track["xAcceleration"][current_index]
    y_acceleration = track["yAcceleration"][current_index]

    # Normalize the center point to the range [0, 1]
    center_point = [center_point[0] / height_width[1], center_point[1] / height_width[0]]

    return {
        "pts": pts,
        "center": center_point,
        "object_class": object_class,
        "heading": heading,
        "xVelocity": x_velocity,
        "yVelocity": y_velocity,
        "xAcceleration": x_acceleration,
        "yAcceleration": y_acceleration
    }

def _extract_historical_data(ego_track, ego_track_meta, scene_id, current_frame, scale_down_factor, tracks, tracks_meta, visibility_data):
    ego_vehicle_track_idx = ego_track["trackId"]
    # iterate through all the historical frames upto the current_frame
    historical_adjacent_obs, historical_ego_obs, map_obs, visible_tracks_pts, last_recorded_t = {}, [], [], [], {}

    backgrond_img = background_images[scene_id]
    starting_frame = current_frame - history_length

    history_t = 0
    for t in range(starting_frame, current_frame + 1):  # T, N, D
        map = deepcopy(backgrond_img)

        # Extract historical observations for the ego vehicle
        ego_track_info = _extract_track_info(ego_track, ego_track_meta, t, backgrond_img.shape, scale_down_factor)
        cv2.fillPoly(map, [ego_track_info["pts"]], (255, 180, 200))
        historical_ego_obs.append(np.array([ego_track_info["center"][0],
                                            ego_track_info["center"][1],
                                            ego_track_info["heading"],
                                            ego_track_info["xVelocity"],
                                            ego_track_info["yVelocity"],
                                            ego_track_info["xAcceleration"],
                                            ego_track_info["yAcceleration"],
                                            class_dict[ego_track_meta["class"]]]))

        visible_track_ids = list(visibility_data[(visibility_data["trackId"] == ego_vehicle_track_idx)
                                                      & (visibility_data["frame"] == t)
                                                      & (visibility_data["visibility"] == True)
                                                      & (visibility_data["recordingId"] == scene_id)
                                                      & (visibility_data["located"] == 'FRONT')]
                                 ["adjacentTrackId"].drop_duplicates())

        # Extract historical observations for visible tracks
        recorded_tack_ids = []
        for track_idx in visible_track_ids:
            track = tracks[(tracks["trackId"] == track_idx)
                                & (tracks["recordingId"] == scene_id)].iloc[0].to_dict()
            track_meta = tracks_meta[(tracks_meta["trackId"] == track_idx)
                                          & (tracks_meta["recordingId"] == scene_id)].iloc[0].to_dict()

            track_info = _extract_track_info(track, track_meta, t, backgrond_img.shape, scale_down_factor)
            cv2.fillPoly(map, [track_info["pts"]], (150, 150, 50))  # 150, 100, 150

            # Calculate distance from the ego vehicle to the track center
            distance_ego = math.sqrt((track_info["center"][0] - ego_track_info["center"][0]) ** 2 +
                                     (track_info["center"][1] - ego_track_info["center"][1]) ** 2)

            track_data = np.array([track_info["center"][0], track_info["center"][1], track_info["heading"],
                                   track_info["xVelocity"], track_info["yVelocity"], track_info["xAcceleration"],
                                   track_info["yAcceleration"], class_dict[track_meta["class"]],
                                   (history_t / history_length), distance_ego])

            last_recorded_t[track_idx] = history_t  # Store the last recorded time for the track

            # visible track points
            if t == current_frame:
                visible_tracks_pts.append(np.squeeze(track_info["pts"]))

            if track_idx not in historical_adjacent_obs.keys():
                if t > starting_frame:
                    # This object appeared lately. So have to add nulls/empty/zeros for previous frames
                    historical_adjacent_obs[track_idx] = [np.zeros(num_features + 2) for _ in
                                                          range(starting_frame, t)]  # +1 for distance and class
                    historical_adjacent_obs[track_idx].append(track_data)
                else:
                    historical_adjacent_obs[track_idx] = [track_data]
            else:
                historical_adjacent_obs[track_idx].append(track_data)

            recorded_tack_ids.append(track_idx)

        # Fill the historical observations with zeros for the tracks that are not visible in the current frame
        for track_idx in historical_adjacent_obs.keys():
            if track_idx not in recorded_tack_ids:
                historical_adjacent_obs[track_idx].append(np.zeros(num_features + 2))  # +1 for distance and class

        # map_resized = cv2.resize(map, (224, 224), interpolation=cv2.INTER_AREA)
        # map_obs.append(map_resized)

        history_t += 1

    return historical_adjacent_obs, historical_ego_obs, map_obs, visible_tracks_pts, last_recorded_t


def _extract_ground_truth_data(ego_track, ego_track_meta, scene_id, current_frame, scale_down_factor, tracks, tracks_meta, background_images):
    ego_vehicle_track_idx = ego_track["trackId"]

    hidden_tracks_visibility_df = visibility_data[(visibility_data["frame"] == current_frame)
                                                       & (visibility_data["trackId"] == ego_vehicle_track_idx)
                                                       & (visibility_data["recordingId"] == scene_id)
                                                       & (visibility_data["visibility"] == False)
                                                       & (visibility_data["located"] == 'FRONT')]

    hidden_track_idx = hidden_tracks_visibility_df["adjacentTrackId"].drop_duplicates().tolist()
    hidden_tracks = tracks[(tracks["trackId"].isin(hidden_track_idx))
                                & (tracks["recordingId"] == scene_id)].to_dict('records')

    ego_track_info = _extract_track_info(ego_track, ego_track_meta, current_frame,
                                              background_images[scene_id].shape, scale_down_factor)

    hidden_tracks_pts = []
    hidden_tracks_ids = []
    for track in hidden_tracks:
        track_meta = tracks_meta[(tracks_meta["trackId"] == track["trackId"])
                                      & (tracks_meta["recordingId"] == scene_id)].iloc[0].to_dict()
        track_info = _extract_track_info(track, track_meta, current_frame,
                                              background_images[scene_id].shape, scale_down_factor)

        # Calculate distance from the ego vehicle to the track center
        distance = math.sqrt((track_info["center"][0] - ego_track_info["center"][0]) ** 2 +
                             (track_info["center"][1] - ego_track_info["center"][1]) ** 2)
        pts = np.squeeze(track_info["pts"])

        # Create a numpy array with the track information
        track_data = np.array([track_info["center"][0], track_info["center"][1], track_info["heading"],
                               track_info["xVelocity"], track_info["yVelocity"], track_info["xAcceleration"],
                               track_info["yAcceleration"], class_dict[track_meta["class"]], distance])

        hidden_tracks_pts.append(np.squeeze(pts))
        hidden_tracks_ids.append(track["trackId"])

        # cv2.fillPoly(backgrond_img, [pts], (0, 255, 255))

    # cv2.imshow('Image with Polygon', backgrond_img)
    return hidden_tracks_pts, hidden_tracks_ids


def _extract_edge_info( historical_adjacent_obs, hidden_ogm_cells, last_recorded_t) -> Tuple[list, list]:
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


def create_args():
    cs = argparse.ArgumentParser(description="Dataset Tracks Visualizer")
    # --- Input ---
    cs.add_argument('--dataset_dir', default="../data/",
                    help="Path to directory that contains the dataset csv files.", type=str)
    cs.add_argument('--annotations_dir', default="../data/annotations/",
                    help="Path to directory that contains the dataset csv files.", type=str)
    cs.add_argument('--history_length', default=20,
                    help="Number of previous timesetps that includes in the historical observations of a data entry",
                    type=int)
    cs.add_argument('--start_recording_id', default=32,
                    help="Starting recording id that visibility data extraction starts from", type=int)
    cs.add_argument('--end_recording_id', default=32,
                    help="Final recording id that visibility data extraction ends from", type=int)

    return vars(cs.parse_args())


if __name__ == '__main__':
    config = create_args()

    scene_ids = set()
    for fname in os.listdir(config['dataset_dir']):
        match = re.match(r"(\d+)_.*\.csv", fname)
        if match:
            scene_ids.add(match.group(1))

    scene_ids = sorted(scene_ids)
    print(f"Found scenes: {scene_ids}")

    input_path = config['dataset_dir']

    tracks = []
    tracks_meta = []
    visibility_data = []
    background_images = {}
    fixed_blocks_info = {}
    frame_to_track_idxs = {}
    class_dict = {'car': 0, 'truck_bus': 1, 'bicycle': 2, 'pedestrian': 3}

    start_scene = config["start_recording_id"]
    end_scene = config["end_recording_id"]

    filename = f"observations_{start_scene}_{end_scene}.pkl"
    observation_data_path = os.path.join(input_path, filename)

    data_dict = {}

    for scene_id in scene_ids:

        if int(scene_id) < start_scene or int(scene_id) > end_scene:
            continue

        tracks_file = os.path.join(input_path, f"{scene_id}_tracks.csv")
        tracks_meta_file = os.path.join(input_path, f"{scene_id}_tracksMeta.csv")
        recording_meta_file = os.path.join(input_path, f"{scene_id}_recordingMeta.csv")
        fixed_blocks_file = os.path.join(input_path, f"{scene_id}_fixedBlocks.csv")
        visibility_file = os.path.join(input_path, f"{scene_id}_visibilityData.csv")

        tracks_, tracks_meta_, recording_meta_, fixed_blocks_info_ = read_from_csv(
            tracks_file, tracks_meta_file, recording_meta_file, fixed_blocks_file, include_px_coordinates=True
        )

        visibility_df = pd.read_csv(visibility_file)
        fixed_blocks_info[int(scene_id)] = fixed_blocks_info_

        # Collect DataFrames
        tracks.append(pd.DataFrame(tracks_))
        tracks_meta.append(pd.DataFrame(tracks_meta_))
        visibility_data.append(visibility_df)

        # Store background images for scenes
        bg_path = os.path.join(input_path, 'semantic_maps', f"{scene_id}_background.png")
        img = cv2.imread(bg_path)
        background_images[int(scene_id)] = img

    input_path = config["dataset_dir"]
    dataset = 'ind'
    history_length = config["history_length"]
    num_features = 8  # x, y, heading, xVelocity, yVelocity, xAcceleration, yAcceleration, t

    # Load dataset specific visualization parameters from file
    dataset_params_path = Path(os.path.join(input_path, 'visualizer_params')) / "visualizer_params.json"

    with open(dataset_params_path) as f:
        dataset_params = json.load(f)

    dataset_params = dataset_params["datasets"][dataset]
    scale_down_factor = dataset_params["scale_down_factor"]

    tracks = pd.concat(tracks, ignore_index=True)
    tracks_meta = pd.concat(tracks_meta, ignore_index=True)
    visibility_data = pd.concat(visibility_data, ignore_index=True)

    # read all annotation filenames
    p = Path(config['annotations_dir'])
    keys = [item.name.split(".")[0] for item in p.iterdir() if item.is_file()]
    keys = list(set(keys))  # drop duplicates if any

    scene_dict = dict.fromkeys([int(scene_id) for scene_id in scene_ids], {})
    scene_dict_keys = list(scene_dict.keys())

    for key in keys:
        key_elements = key.split("_")
        scene_id = int(key_elements[0])
        frame = int(key_elements[1])
        ego = int(key_elements[2])
        if len(scene_dict[scene_id]) == 0:
            scene_dict[scene_id] = {frame: ego}
        else:
            scene_dict[scene_id][frame] = ego

    for scene_id in scene_ids:
        scene_id = int(scene_id)
        if scene_id < start_scene or scene_id > end_scene:
            continue

        tracks_meta = tracks_meta[(tracks_meta["recordingId"] == scene_id)]
        visibility_data = visibility_data[(visibility_data["recordingId"] == scene_id)]
        tracks = tracks[(tracks["recordingId"] == scene_id)]

        # get the frames for the current scene
        frames = list(scene_dict[scene_id].keys())
        frames.sort()

        # Create a mapping between frame and idxs of tracks for quick lookup during playback
        frame_to_track_idxs = {}
        for i_frame in frames:
            indices = \
                tracks_meta[(tracks_meta["initialFrame"] <= i_frame) & (tracks_meta["finalFrame"] >= i_frame)][
                    "trackId"].tolist()
            frame_to_track_idxs[i_frame] = indices

        # We have to find out timesteps that have atleast one hidden record in the visiblity data.
        # We cannot start from the minimum_frame as we have to include the history as well.
        for i_frame in frames:
            # for history_frame in range(i_frame - history_length - 1, i_frame + 1):

            # Extract ego vehicle track information
            ego_vehicle_track_idx = scene_dict[scene_id][i_frame]
            ego_track = tracks[(tracks["trackId"] == ego_vehicle_track_idx)
                                    & (tracks["recordingId"] == scene_id)].iloc[0].to_dict()
            ego_track_meta = tracks_meta[(tracks_meta["trackId"] == ego_vehicle_track_idx)
                                              & (tracks_meta["recordingId"] == scene_id)].iloc[0].to_dict()

            pts_ego = _extract_track_info(ego_track, ego_track_meta, i_frame,
                                               background_images[scene_id].shape, scale_down_factor)["pts"]
            heading_ego = _get_heading(ego_track, ego_track_meta, i_frame)

            # extract historical data for the ego ddd
            historical_adjacent_obs, historical_ego_obs, map_obs, visible_tracks_pts, last_recorded_t = (
                _extract_historical_data(ego_track, ego_track_meta, scene_id, i_frame, scale_down_factor, tracks, tracks_meta, visibility_data))

            # extract ground truth data for the ego vehicle
            hidden_tracks_pts, hidden_tracks_ids = _extract_ground_truth_data(ego_track, ego_track_meta, scene_id,
                                                                              i_frame, scale_down_factor,
                                                                              tracks, tracks_meta, background_images)

            # Create OGM
            ogm, ogm_gt, hidden_ogm_cells, hidden_cell_polygon_xys, cell_coords, hidden_tracks_id_ogm = create_OGM_ego(
                pts_ego.squeeze(),
                heading_ego,
                visible_tracks_pts,
                hidden_tracks_pts,
                hidden_tracks_ids,
                background_images[
                    scene_id],
                fixed_blocks_info[
                    scene_id])

            if ogm is None:
                continue

            # Check if there are any hidden ogm cells. Should have at least one hidden ogm cell to create a valid sample
            if len(hidden_ogm_cells) == 0:
                continue

            # Extract distances for hidden ogm cells from adjacent tracks (This is a bi-partition graph)
            edge_weights, edge_index = _extract_edge_info(historical_adjacent_obs, hidden_ogm_cells,
                                                               last_recorded_t)

            data_dict[str(scene_id) + '_' + str(i_frame) + '_' + str(ego_vehicle_track_idx)] = {
                "historical_adjacent_obs": historical_adjacent_obs,
                "historical_ego_obs": historical_ego_obs,
                "map_obs": None,
                "hidden_tracks_pts": hidden_tracks_pts,
                "visible_tracks_pts": visible_tracks_pts,
                "edge_weights": edge_weights,
                "edge_index": edge_index,
                "ogm": ogm,
                "ogm_gt": ogm_gt,
                "hidden_ogm_cells": hidden_ogm_cells,
                "hidden_cell_polygon_xys": hidden_cell_polygon_xys,
                "cell_coords": cell_coords,
                "hidden_tracks_id_ogm": hidden_tracks_id_ogm
            }

            print(str(scene_id) + '_' + str(i_frame) + '_' + str(ego_vehicle_track_idx))

    pickle.dump(data_dict, open(observation_data_path, "wb"))
