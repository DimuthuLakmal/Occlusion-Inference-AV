import cv2
import numpy as np
from pathlib import Path
import re
import os
import math

from loguru import logger
import pickle

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def _rot2d(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])

def get_vert(x, y, heading, length=10.0, width=10.0):
    """
    Returns Nx2 array of polygon vertices for a rectangle centered at (x, y)
    rotated by `heading` (radians). length/width are in the same units as x,y.
    """
    # rectangle corners in the vehicle's local frame (centered at origin)
    L, W = length, width
    local = np.array([
        [+L / 2, +W / 2],
        [+L / 2, -W / 2],
        [-L / 2, -W / 2],
        [-L / 2, +W / 2],
    ])

    R = _rot2d(heading)
    return (local @ R.T) + np.array([x, y])

def draw_cells(ax, x, y, heading, cell_size=20, as_center=True, color=None):
    """
    cells: list of tuples (x, y, is_black) where is_black ∈ {0,1}
           x,y in the same pixel coord system as your map.
    cell_size: side length in pixels.
    as_center: True if (x,y) is the cell center; False if it's top-left.
    """

    verts = get_vert(x, y, heading, length=cell_size, width=cell_size)

    ax.add_patch(patches.Polygon(verts, closed=True,
                                 facecolor=color))


def draw_poly(ax, vehicle, color, timestep, map_img=None):
    x, y, heading, ax_, ay_ = vehicle[timestep][0], vehicle[timestep][1], vehicle[timestep][2], vehicle[timestep][
            3], vehicle[timestep][4]
    x = x * map_img.shape[1]
    y = y * map_img.shape[0]
    verts = get_vert(x, y, heading)
    ax.add_patch(patches.Polygon(verts, closed=True, facecolor=color,
                                     edgecolor=color, linewidth=0.8))


def draw_circle(vehicle, color, map_img=None):
    x, y, heading, vehicle_type = vehicle[0], vehicle[1], vehicle[2], vehicle[7]

    if vehicle_type == 2.0:  # Bycle
        color = (221, 160, 221)
    elif vehicle_type == 3.0:  # Pedestrian
        color = (128, 0, 128)

    # cv polygon
    return cv2.circle(map_img, (int(x), int(y)), 10, color, -1)


def draw_frame(map_img, adj_vehicle_data, ego_vehicle_data, cells, targets, masks, losses, timestep, num_timesteps, save_path=None):

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(map_img)

    # Draw cells ONLY on the last timestep
    if (cells is not None) and (timestep == num_timesteps - 1):
        for i, (cell, target, mask) in enumerate(zip(cells, targets, masks)):
            x = cell[0] * map_img.shape[1]
            y = cell[1] * map_img.shape[0]
            label = target
            if mask and losses is None:
                color = 'black' if label == 1 else 'white'
                draw_cells(ax, x, y, ego_vehicle_data[timestep][2], cell_size=20, color=color)
            elif mask and losses is not None:
                loss_color = losses[i]/2
                if loss_color > 1:
                    loss_color = 1
                color = (loss_color, 0, 0, 1)
                draw_cells(ax, x, y, ego_vehicle_data[timestep][2], cell_size=20, color=color)


    for vehicle in adj_vehicle_data:
        draw_poly(ax, vehicle, 'brown', timestep, map_img)

    draw_poly(ax, ego_vehicle_data, 'black', timestep, map_img)

    ax.set_axis_off()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    else:
        plt.show()

    print("Frame drawn")


def convert_to_images(keys, data_dict, background_images, targets, masks, losses):
    for i, (key, target, mask, loss) in enumerate(zip(keys, targets, masks, losses)):
        scene_id = int(key[0])
        current_frame = int(key[1])
        ego_vehicle_track_idx = int(key[2])

        # map
        backgrond_img = background_images[i]

        # Historical observations
        historical_adjacent_obs, historical_ego_obs = data_dict["historical_adjacent_obs"][i], data_dict[
            "historical_ego_obs"][i]
        hidden_ogm_cells = data_dict["hidden_ogm_cells"][i]

        num_timesteps = 21

        historical_adjacent_obs = historical_adjacent_obs.detach().cpu().numpy()
        historical_ego_obs = historical_ego_obs.detach().cpu().numpy()
        hidden_ogm_cells = hidden_ogm_cells.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()
        loss = loss.detach().cpu().numpy()

        historical_adjacent_obs[:, :, 2:3] = historical_adjacent_obs[:, :,
                                              2:3] * 360.0  # Normalize heading to [0, 1]. This is a mistake done when extracting the data
        historical_adjacent_obs[:, :, 3:] = historical_adjacent_obs[:, :, 3:] * 10.0

        for t in range(num_timesteps):
            draw_frame(backgrond_img, historical_adjacent_obs, historical_ego_obs, hidden_ogm_cells, target, mask, None, t,
                       num_timesteps)
            if t == num_timesteps - 1:
                draw_frame(backgrond_img, historical_adjacent_obs, historical_ego_obs, hidden_ogm_cells, target, mask,
                           loss, t, num_timesteps)

        # image_folder = 'frames'
        # video_name = 'vehicle_animation.mp4'
        # frame_rate = 5  # fps
        #
        #
        # frame_array = []
        # for t in range(num_timesteps):
        #     filename = f"{image_folder}/frame_{t:03d}.png"
        #     img = cv2.imread(filename)
        #     height, width, _ = img.shape
        #     frame_array.append(img)
        #
        # out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))
        #
        # for frame in frame_array:
        #     out.write(frame)
        # out.release()


def interactive_playback(image_folder, total_frames):
    fig, ax = plt.subplots()
    img = mpimg.imread(f"{image_folder}/frame_000.png")
    im = ax.imshow(img)
    ax.set_title("Use arrow keys to scroll frames")

    def on_key(event):
        nonlocal frame_idx
        if event.key == 'right':
            frame_idx = (frame_idx + 1) % total_frames
        elif event.key == 'left':
            frame_idx = (frame_idx - 1) % total_frames
        img = mpimg.imread(f"{image_folder}/frame_{frame_idx:03d}.png")
        im.set_data(img)
        fig.canvas.draw_idle()

    frame_idx = 0
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


# Run interactive viewer
# interactive_playback("frames", num_timesteps)
