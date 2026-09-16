import pandas as pd
import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt


def create_dataframe_chunk(recording_ids, track_ids, frames, visible_track_ids):
    """
    Creates a DataFrame chunk with columns: recordingId, trackId, frame, visibleTrackId.
    Each parameter is a list corresponding to multiple rows.
    """
    data = {
        'recordingId': recording_ids,
        'trackId': track_ids,
        'frame': frames,
        'visibleTrackId': visible_track_ids
    }
    return pd.DataFrame(data)

def heading_to_vector(heading_deg):
    theta = np.radians(-heading_deg)  # CW → CCW
    return np.array([np.cos(theta), np.sin(theta)])


def classify_corners(corners, heading_deg):
    """
    Classify unordered corners into FL, FR, RR, RL given heading.
    """
    center = np.mean(corners, axis=0)
    forward = heading_to_vector(heading_deg)
    right = np.array([forward[1], -forward[0]])

    labels = {}
    for corner in corners:
        vec = np.array(corner) - center
        fwd_proj = np.dot(vec, forward)
        right_proj = np.dot(vec, right)

        if fwd_proj > 0 and right_proj < 0:
            labels['FL'] = tuple(corner)
        elif fwd_proj > 0 and right_proj > 0:
            labels['FR'] = tuple(corner)
        elif fwd_proj < 0 and right_proj > 0:
            labels['RR'] = tuple(corner)
        elif fwd_proj < 0 and right_proj < 0:
            labels['RL'] = tuple(corner)

    return labels['FL'], labels['FR'], labels['RR'], labels['RL']


def get_driver_center(corners, heading_deg):
    FL, FR, RR, RL = classify_corners(corners, heading_deg)

    # Compute vehicle centerline
    rear_center = (np.array(RL) + np.array(RR)) / 2
    front_center = (np.array(FL) + np.array(FR)) / 2

    # Position of rear-view mirror base: a bit forward from rear
    driver_seat_ratio = 0.75
    if type == "car":
        driver_seat_ratio = 0.75
    elif type == "truck_bus":
        driver_seat_ratio = 0.9

    driver_seat = rear_center + driver_seat_ratio * (front_center - rear_center)
    return driver_seat


def get_rear_fov_polygons(corners, heading_deg, fov_length=20.0, fov_angle_passenger=30, fov_angle_driver=37.6, type=None):
    """
    Compute driver, passenger, and rear-view mirror FOVs as polygons.
    """

    def create_side_wedge(mirror_pos, along_vec, outward_angle_deg, length):
        base_angle = np.arctan2(along_vec[1], along_vec[0])
        outward_angle = base_angle + np.radians(outward_angle_deg)

        along_point = mirror_pos + length * np.array([np.cos(base_angle), np.sin(base_angle)])
        outward_point = mirror_pos + length * np.array([np.cos(outward_angle), np.sin(outward_angle)])

        return Polygon([mirror_pos, along_point, outward_point])

    FL, FR, RR, RL = classify_corners(corners, heading_deg)
    forward = heading_to_vector(heading_deg)
    right_vec = np.array([forward[1], -forward[0]])

    mirror_base_center = get_driver_center(corners, heading_deg)
    half_width = np.linalg.norm(np.array(FR) - np.array(FL)) / 2 # slightly narrower than full width

    # Driver & passenger mirror positions
    driver_mirror = mirror_base_center + right_vec * half_width
    passenger_mirror = mirror_base_center - right_vec * half_width

    # Driver side wedge
    driver_side_vec = np.array(RR) - np.array(FR)
    driver_poly = create_side_wedge(driver_mirror, driver_side_vec, fov_angle_driver, fov_length)

    # Passenger side wedge
    passenger_side_vec = np.array(RL) - np.array(FL)
    passenger_poly = create_side_wedge(passenger_mirror, passenger_side_vec, -fov_angle_passenger, fov_length)

    # Rear-view mirror polygon: rear edge RL→RR extended backward
    RL_arr = np.array(RL)
    RR_arr = np.array(RR)
    backward_vec = -forward

    RL_far = RL_arr + backward_vec * fov_length
    RR_far = RR_arr + backward_vec * fov_length

    rear_poly = Polygon([RL_arr, RR_arr, RR_far, RL_far])

    return driver_poly, passenger_poly, rear_poly
