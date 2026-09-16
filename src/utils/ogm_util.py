import random
from copy import deepcopy

import numpy as np
import cv2
from shapely.geometry import LineString
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

from utils.utils import get_driver_center

# ---- CONFIGURATION ----
CELL_SIZE = 20
GRID_ROWS = GRID_COLS = 20  # 20x20 grid


def create_OGM_ego(ego_pts, ego_heading, visible_bbox, hidden_bbox, image, fixed_blocks):
    driver_seat_loc = get_driver_center(ego_pts, ego_heading)

    # Example ego position (in pixel coordinates)
    # ego vehicle info
    # ego_x, ego_y = 850, 900
    # ego_heading_deg = 45  # example: 45°

    ego_heading_rad = np.deg2rad(-1 * ego_heading)

    visible_vehicle_polygons = [Polygon(bbox) for bbox in visible_bbox]  # list of polygons for other vehicles
    hidden_vehicle_polygons = [Polygon(bbox) for bbox in hidden_bbox]  # list of polygons for other vehicles
    fixed_blocks_polygons = [Polygon(bbox) for bbox in fixed_blocks]  # list of polygons for other vehicles

    # prepare transformation
    cos_theta = np.cos(ego_heading_rad)
    sin_theta = np.sin(ego_heading_rad)

    R = np.array([[cos_theta, -sin_theta],
                  [sin_theta,  cos_theta]])

    # origin of the grid is just in front of ego — let’s keep it at ego for simplicity
    origin = np.array([driver_seat_loc[0], driver_seat_loc[1]])

    # construct cell polygons in ego frame and transform to global frame
    cell_polygons = [[None for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]

    # ---- BUILD GRID ----
    grid = np.full((GRID_ROWS, GRID_COLS), 0.5)  # default: 0.5 = unknown

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            # cell position in ego’s local frame
            x_local_min = c * CELL_SIZE
            y_local_min = - (GRID_ROWS // 2 - r) * CELL_SIZE
            x_local_max = x_local_min + CELL_SIZE
            y_local_max = y_local_min + CELL_SIZE

            # corners of cell in ego frame
            corners_local = np.array([
                [x_local_min, y_local_min],
                [x_local_max, y_local_min],
                [x_local_max, y_local_max],
                [x_local_min, y_local_max]
            ])

            # transform to global
            corners_global = (R @ corners_local.T).T + origin

            cell_polygons[r][c] = Polygon(corners_global)

    all_vehicle_polygons = visible_vehicle_polygons + hidden_vehicle_polygons
    known_occupied_polygons = (visible_vehicle_polygons + fixed_blocks_polygons)  # only visible vehicles and fixed blocks are known to be occupied

    def is_fully_occupied(cell, poly):
        intersection = cell.intersection(poly)

        polygon_ratio = poly.area / cell.area
        # following if else check if the cell is significantly larger than the polygon (eg: in case of Humans, treees etc.)

        # if the is comparatively larger polygon, then check if it occupies more than 70% of the cell
        if polygon_ratio > 0.7:
            overlap_ratio = intersection.area / cell.area
            # if cell is smaller and overlap is more than 30%, then it is occupied
            if overlap_ratio > 0.2:
                return True
        else:
            # if the polygon is smaller, then check if the intersection is comparable with the polygon itself.
            overlap_ratio = intersection.area / poly.area
            if overlap_ratio > 0.5:
                return True

        return False

    # ---- OCCUPANCY TEST ----
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            cell = cell_polygons[r][c]
            intersected = False
            for poly in all_vehicle_polygons:
                if cell.intersects(poly):
                    if is_fully_occupied(cell, poly):
                        grid[r, c] = 1
                        intersected = True
                        break
                    else:
                        grid[r, c] = 0

            for poly in fixed_blocks_polygons:
                if cell.intersects(poly):
                    if is_fully_occupied(cell, poly) and grid[r, c] != 1:  # if already occupied by a vehicle, do not override
                        grid[r, c] = 3
                        intersected = True
                        break

            if not intersected:
                grid[r, c] = 0  # free

    grid_gt = deepcopy(grid)  # keep ground truth for occlusion detection

    # Mark if the cell is occupied by a hidden vehicle. This will be used later for loop
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            cell = cell_polygons[r][c]
            for poly in hidden_vehicle_polygons:
                if cell.intersects(poly):
                    if is_fully_occupied(cell, poly):
                        grid[r, c] = 2  # occupied
                        break
                    else:
                        grid[r, c] = 0

    # Check if at least one of the cell is occupied by a hidden vehicle
    # If not, then return None as the data is not useful
    has_hidden_vehicle = np.any(grid == 2)
    if not has_hidden_vehicle:
        return None, None, None, None  # no hidden vehicles detected

    # The following indices will be used to sample the cells later
    occupied_by_hidden_indices = np.where(grid == 2)

    # ---- VISIBILITY TEST ----
    # simple line-of-sight: if a cell center has line blocked by vehicles → occluded
    def is_visible(cell_center, ego_pos, occupied_polygons):
        line = LineString([ego_pos, cell_center])
        for vp in occupied_polygons:
            if vp.intersects(line):
                return False
        return True

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            cx, cy = cell_polygons[r][c].centroid.coords[0]
            if not is_visible((cx, cy), (driver_seat_loc[0], driver_seat_loc[1]), known_occupied_polygons):
                if grid[r, c] == 0:  # only override if free. We don't mark cells occupied by hidden vehicles as hidden yet. Will do it later.
                    grid[r, c] = 0.5  # mark as occluded


    # Filter out irrelevant regions (e.g., black areas in the map)
    irrelevant_mask = _filter_irrelevant_regions(cell_polygons, image)
    grid[irrelevant_mask] = 0.7
    grid_gt[irrelevant_mask] = 0.7

    ### Data Sampling ##########
    unoccupied_indices = np.where(grid == 0.5)

    # Decides which type is more. Usually this is free cells, But we cannot 100% guarantee that.
    num_selections = min(len(occupied_by_hidden_indices[0]), len(unoccupied_indices[0]))
    max_array_size = max(len(occupied_by_hidden_indices[0]), len(unoccupied_indices[0]))
    random_indices = random.sample(range(max_array_size), num_selections)

    selected_indices_unoccupied, selected_indices_occupied = None, None
    if len(occupied_by_hidden_indices[0]) == num_selections:  # if occupied cells are less, then randomly select from unoccupied cells
        selected_indices_unoccupied = (unoccupied_indices[0][random_indices], unoccupied_indices[1][random_indices])
        selected_indices_occupied = occupied_by_hidden_indices
    else:  # if unoccupied cells are less, then randomly select from occupied cells
        selected_indices_occupied = (occupied_by_hidden_indices[0][random_indices], occupied_by_hidden_indices[1][random_indices])
        selected_indices_unoccupied = unoccupied_indices

    selected_indices_raws = np.concatenate((selected_indices_unoccupied[0], selected_indices_occupied[0]))
    selected_indices_cols = np.concatenate((selected_indices_unoccupied[1], selected_indices_occupied[1]))

    # Now we can safely mark cells occupied by hidden vehicles as not visible
    # grid[occupied_by_hidden_indices] = 0.5  # mark as occluded

    # Store hidden cell centroids for extracting edge info in later steps
    hidden_cell_data = []
    hidden_cell_polygon_xys = []
    image_width_height = image.shape[:2]
    for r, c in zip(selected_indices_raws, selected_indices_cols):
        cx, cy = cell_polygons[r][c].centroid.coords[0]
        hidden_cell_data.append(
            [cx / image_width_height[1], cy / image_width_height[0], grid_gt[r, c]])  # normalized x, y, occupancy
        hidden_cell_polygon_xys.append(cell_polygons[r][c].exterior.coords[:])


    # ---- VISUALIZATION ----
    # _visualise(image, visible_vehicle_polygons + hidden_vehicle_polygons, driver_seat_loc, cell_polygons, grid)
    # _visualise(image, visible_vehicle_polygons + hidden_vehicle_polygons, driver_seat_loc, cell_polygons, grid_gt)

    return grid, grid_gt, hidden_cell_data, hidden_cell_polygon_xys


def _filter_irrelevant_regions(cell_polygons, map_image):
    # assumes image is a 3-channel RGB image (numpy array)
    COVERAGE_THRESHOLD = 0.8  # % of pixels that must be black to consider cell as irrelevant

    irrelevant_mask = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            poly = cell_polygons[r][c]

            # Convert polygon to integer pixel mask
            # Step 1: create binary mask for the polygon
            mask = np.zeros(map_image.shape[:2], dtype=np.uint8)
            pts = np.array([np.array(poly.exterior.coords, dtype=np.int32)])
            cv2.fillPoly(mask, pts, 255)

            # Step 2: convert map image to grayscale
            gray = cv2.cvtColor(map_image, cv2.COLOR_BGR2GRAY)

            # Step 3: extract only masked region
            region_pixels = gray[mask == 255]

            # Step 4: compute black pixel ratio
            if region_pixels.size == 0:
                continue  # skip degenerate cells

            black_ratio = np.mean(region_pixels == 0)

            # Step 5: update mask
            if black_ratio > COVERAGE_THRESHOLD:
                irrelevant_mask[r, c] = True

    return irrelevant_mask


def _visualise(image, vehicle_polygons, driver_seat_loc, cell_polygons, grid):
    # ---- VISUALIZE ----
    height, width = image.shape[:2]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect('equal')

    ax.imshow(image)

    # Draw vehicle polygons
    for vp in vehicle_polygons:
        x, y = vp.exterior.xy
        ax.fill(x, y, color='red', alpha=0.5)

    # Draw ego position
    ax.plot(driver_seat_loc[0], driver_seat_loc[1], 'bo', markersize=8, label='Ego')

    # Draw grid
    colors = {0: 'white', 1: 'black', 0.5: 'gray', 0.7: 'orange', 2: 'yellow', 3: 'brown'}
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            cell = cell_polygons[r][c]
            x, y = cell.exterior.xy
            if grid[r, c] == 2:
                print('A 2 is here')
            ax.fill(x, y, color=colors[grid[r, c]], alpha=0.7, edgecolor='k', linewidth=0.2)

    plt.title("Occupancy Grid Map")
    plt.legend()
    plt.show()


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
    a = np.deg2rad(heading)
    R = _rot2d(a)
    return (local @ R.T) + np.array([x, y])