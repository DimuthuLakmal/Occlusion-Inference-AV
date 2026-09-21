import math
import json
import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List
from loguru import logger
import visilibity as vis
from shapely.geometry import Polygon as ShapelyPolygon
import pandas as pd

from src.utils.utils import get_rear_fov_polygons, get_driver_center


class VisibilityChecker(object):
    def __init__(self, config: dict, tracks: List[dict], tracks_meta: List[dict], recording_meta: dict, fixed_blocks_info: List[List]=None, csv_file: str = None):
        self.config = config
        self.input_path = config["dataset_dir"]
        self.dataset = 'ind'
        self.location_id = recording_meta["locationId"]
        self.fixed_blocks_info = fixed_blocks_info
        self.csv_file = csv_file

        # Currently clicked vehicle
        self.clicked_track_id = None
        # Color mapping for surrounding vehicles
        self.surrounding_vehicles_colors = {
            "leadId": "red",
            "rightLeadId": "orange",
            "rightAlongsideId": "black",
            "rightRearId": "purple",
            "rearId": "blue",
            "leftRearId": "green",
            "leftAlongsideId": "brown",
            "leftLeadId": "yellow"
        }
        vehicle_keys = list(self.surrounding_vehicles_colors.keys())
        self.surrounding_vehicles_ids = dict(zip(vehicle_keys, -1 * np.ones(len(vehicle_keys), dtype=int)))

        # Load dataset specific visualization parameters from file
        dataset_params_path = Path(os.path.join(self.input_path, 'visualizer_params')) / "visualizer_params.json"

        if not dataset_params_path.exists():
            logger.error("Could not find dataset visualization parameters in {}", dataset_params_path)
            sys.exit(-1)

        with open(dataset_params_path) as f:
            self.dataset_params = json.load(f)

        self.dataset_params = self.dataset_params["datasets"][self.dataset]
        self.scale_down_factor = self.dataset_params["scale_down_factor"]

        self.tracks = tracks
        self.tracks_meta = tracks_meta
        self.recording_meta = recording_meta

        # Check whether tracks and tracks_meta match each other
        error_message = "The tracks file and the tracksMeta file is not matching each other. " \
                        "Please check whether you modified any of these files."
        if len(tracks) != len(tracks_meta):
            logger.error(error_message)
            raise DataError("Failed", error_message)
        for track, track_meta in zip(tracks, tracks_meta):
            if track["trackId"] != track_meta["trackId"]:
                logger.error(error_message)
                raise DataError("Failed", error_message)

        # Determine the first and last frame
        self.minimum_frame = min(meta["initialFrame"] for meta in self.tracks_meta)
        self.maximum_frame = max(meta["finalFrame"] for meta in self.tracks_meta)
        logger.info("The recording contains tracks from frame {} to {}.", self.minimum_frame, self.maximum_frame)

        # Create a mapping between frame and idxs of tracks for quick lookup during playback
        self.frame_to_track_idxs = {}
        for i_frame in range(self.minimum_frame, self.maximum_frame + 1):
            indices = [i_track for i_track, track_meta in enumerate(self.tracks_meta)
                       if track_meta["initialFrame"] <= i_frame <= track_meta["finalFrame"]]
            self.frame_to_track_idxs[i_frame] = indices

        # Show background image
        background_image_path = self.config["background_image_path"]
        if background_image_path and os.path.exists(background_image_path):
            logger.info("Loading background image from {}", background_image_path)
            self.background_image = cv2.cvtColor(cv2.imread(background_image_path), cv2.COLOR_BGR2RGB)
            (self.image_height, self.image_width) = self.background_image.shape[:2]
        else:
            logger.warning("No background image given or path not valid. Using fallback black background.")
            self.image_height, self.image_width = 1700, 1700
            self.background_image = np.zeros((self.image_height, self.image_width, 3), dtype="uint8")

        # Initialize visualization options
        self.current_frame = self.minimum_frame
        self.bb_boxes = {}
        self.frames_written = []


    def extract_visibility(self):
        """
        Main function to draw all tracks and selected annotations for the current frame.
        :param args: Should be unused if called manually. If called by FuncAnimation, args contains a call counter.
        :return: List of artist handles that have been updated. Needed for blitting.
        """

        while self.current_frame < self.maximum_frame:
            # if self.current_frame < 21851:
            #     self.current_frame += 1
            #     continue

            bb_boxes = []
            print(self.current_frame)
            for track_idx in self.frame_to_track_idxs[self.current_frame]:
                track = self.tracks[track_idx]
                track_meta = self.tracks_meta[track_idx]
                initial_frame = track_meta["initialFrame"]
                current_index = self.current_frame - initial_frame

                if track["bboxVis"] is not None:
                    bounding_box = track["bboxVis"][current_index] / self.scale_down_factor
                else:
                    bounding_box = None
                center_points = track["centerVis"] / self.scale_down_factor
                center_point = center_points[current_index]

                if bounding_box is not None:
                    bbox = plt.Polygon(bounding_box, True)
                else:
                    x, y = center_point
                    square_coords = [
                        (x - 1, y - 1),
                        (x + 1, y - 1),
                        (x + 1, y + 1),
                        (x - 1, y + 1),
                        (x - 1, y - 1)
                    ]

                    bbox = plt.Polygon(square_coords, closed=True)

                bbox.center = center_point
                bbox.heading = track["heading"][current_index]
                bbox.type = track_meta["class"]

                # Make bbox clickable to open track info window
                bbox.track_id = track["trackId"]

                bb_boxes.append(bbox)

            # Check the visibility of each track and save the visibility data into a dataframe
            if self.current_frame not in self.frames_written:
                self.frames_written.append(self.current_frame)
                for i, track_idx in enumerate(self.frame_to_track_idxs[self.current_frame]):
                    track = self.tracks[track_idx]
                    track_id = track["trackId"]
                    track_meta = self.tracks_meta[track_idx]

                    if track_meta["class"] == "car" or track_meta["class"] == "truck_bus":
                        # Check visibility of the track
                        data = self._find_visibility(bb_boxes, track_id)
                        visibility_data = data['visibility_data']

                        chunk_df = pd.DataFrame(visibility_data)
                        chunk_df.to_csv(self.csv_file, mode='a', header=False, index=False)

            self.current_frame += 1

        new_headers = ['recordingId', 'trackId', 'adjacentTrackId', 'frame', 'visibility', 'located']
        df = pd.read_csv(self.csv_file, header=None)
        df.columns = new_headers
        df.to_csv(self.csv_file, index=False)


    def _find_visibility(self, all_bb_boxes, track_id):

        if all_bb_boxes == None:
            all_bb_boxes = self.bb_boxes[self.current_frame]

        epsilon = 0.0000001

        # Define the points which will be the outer boundary of the environment
        # Must be COUNTER-CLOCK-WISE(ccw)
        p1 = vis.Point(0, 0)
        p2 = vis.Point(self.image_width, 0)
        p3 = vis.Point(self.image_width, self.image_height)
        p4 = vis.Point(0, self.image_height)

        # Load the values of the outer boundary polygon in order to draw it later
        wall_x = [p1.x(), p2.x(), p3.x(), p4.x(), p1.x()]
        wall_y = [p1.y(), p2.y(), p3.y(), p4.y(), p1.y()]

        # Outer boundary polygon must be COUNTER-CLOCK-WISE(ccw)
        # Create the outer boundary polygon
        walls = vis.Polygon([p1, p2, p3, p4])
        holes = [walls]

        # set observer position
        heading = None
        other_bboxes = []
        ego_points = None
        ego_bbox = None
        type = None

        for i, bb_box in enumerate(all_bb_boxes):
            if bb_box.track_id == track_id:
                ego_points = np.array([[int(xy[0]), int(xy[1])] for xy in bb_box.xy[:-1]])
                ego_bbox = bb_box
                heading = bb_box.heading
                type = bb_box.type
            else:
                points = [vis.Point(int(xy[0]), int(xy[1])) for xy in bb_box.xy[:-1]]
                hole = vis.Polygon(points)
                holes.append(hole)

                other_bboxes.append(bb_box)

        # Creating barriers on the FOV using fixed blocks in the scene (eg: Buildings)
        x_block = []
        y_block = []
        for i, block in enumerate(self.fixed_blocks_info):
            x_block.append([int(xy[0]) for xy in block])
            y_block.append([int(xy[1]) for xy in block])

            points = [vis.Point(int(xy[0]), int(xy[1])) for xy in block[:-1]]
            hole = vis.Polygon(points)
            holes.append(hole)

        driver_seat_loc = get_driver_center(ego_points, heading)
        observer = vis.Point(int(driver_seat_loc[0]), int(driver_seat_loc[1]))

        env = vis.Environment(holes)

        # Necesary to generate the visibility polygon
        observer.snap_to_boundary_of(env, epsilon)
        observer.snap_to_vertices_of(env, epsilon)

        # Obtain the visibility polygon of the 'observer' in the environment
        # previously define
        isovist = vis.Visibility_Polygon(observer, env, epsilon)

        #### Check if the other bbboxes other than the observer are in the visibility polygon
        # First create the visibility polygon of the observer (front and rear view)
        front_view = self._create_cone([driver_seat_loc[0], driver_seat_loc[1]], self.image_width, int(-1 * heading), 90, 3)
        front_view_x, front_view_y = self._save_print(front_view)
        front_view_x.append(front_view_x[0])
        front_view_y.append(front_view_y[0])
        # Define the front view polygon (container)
        front_view_pol = ShapelyPolygon([(x, y) for x, y in zip(front_view_x, front_view_y)])

        # Rearview polygons (View from side mirrors and rearview mirror)
        driver_poly, passenger_poly, rear_poly = get_rear_fov_polygons(ego_points, heading, fov_length=self.image_width, type=type)

        # Check all other bounding boxes for visibility
        visible_front_bboxes, visible_rear_bboxes = [], []
        hidden_front_bboxes, hidden_back_bboxes = [], []
        x_rear_fov, x_front_fov, x_hidden = [], [], []
        y_rear_fov, y_front_fov, y_hidden = [], [], []
        for bb_box in other_bboxes:
            direct_line_of_sight = False
            points = [vis.Point(int(xy[0]), int(xy[1])) for xy in bb_box.xy[:-1]]
            xs, ys = [], []
            for point in points:
                # Check if the point is inside the visibility polygon
                if point._in(isovist, epsilon):
                    xs.append([driver_seat_loc[0], point.x()])
                    ys.append([driver_seat_loc[1], point.y()])

                    direct_line_of_sight = True

            # Further check if the point is inside the front or rearview polygons
            # Define the inner polygon (to test if inside the outer)
            inner = ShapelyPolygon([(point.x(), point.y()) for point in points])
            # Check if inner intersects the rear fov
            in_rear_fov = (inner.intersects(passenger_poly) or inner.intersects(driver_poly)
                           or inner.intersects(rear_poly))
            # Check if inner intersects with front fov
            in_front_fov = inner.intersects(front_view_pol)

            if in_rear_fov and direct_line_of_sight:
                x_rear_fov.append([int(x[0]) for x in bb_box.xy[:, 0:1]])
                y_rear_fov.append([int(y[0]) for y in bb_box.xy[:, 1:2]])
                visible_rear_bboxes.append(bb_box)
            elif in_front_fov and direct_line_of_sight:
                x_front_fov.append([int(x[0]) for x in bb_box.xy[:, 0:1]])
                y_front_fov.append([int(y[0]) for y in bb_box.xy[:, 1:2]])
                visible_front_bboxes.append(bb_box)
            else:
                x_hidden.append([int(x[0]) for x in bb_box.xy[:, 0:1]])
                y_hidden.append([int(y[0]) for y in bb_box.xy[:, 1:2]])

                if in_front_fov:
                    hidden_front_bboxes.append(bb_box)
                else:
                    hidden_back_bboxes.append(bb_box)

        # Saving data into a dataframe with columns: recordingId, trackId, adjacentTrackId, frame, visibility
        visibility_data = []
        for bb_box in visible_rear_bboxes:
            visibility_data.append({
                "recordingId": self.recording_meta["recordingId"],
                "trackId": track_id,
                "adjacentTrackId": bb_box.track_id,
                "frame": self.current_frame,
                "visibility": True,
                "located": "REAR"
            })

        for bb_box in visible_front_bboxes:
            visibility_data.append({
                "recordingId": self.recording_meta["recordingId"],
                "trackId": track_id,
                "adjacentTrackId": bb_box.track_id,
                "frame": self.current_frame,
                "visibility": True,
                "located": "FRONT"
            })

        for bb_box in hidden_front_bboxes:
            visibility_data.append({
                "recordingId": self.recording_meta["recordingId"],
                "trackId": track_id,
                "adjacentTrackId": bb_box.track_id,
                "frame": self.current_frame,
                "visibility": False,
                "located": "FRONT"
            })

        for bb_box in hidden_back_bboxes:
            visibility_data.append({
                "recordingId": self.recording_meta["recordingId"],
                "trackId": track_id,
                "adjacentTrackId": bb_box.track_id,
                "frame": self.current_frame,
                "visibility": False,
                "located": "REAR"
            })

        return {
            'x_front_fov': x_front_fov,
            'y_front_fov': y_front_fov,
            'x_rear_fov': x_rear_fov,
            'y_rear_fov': y_rear_fov,
            'x_hidden': x_hidden,
            'y_hidden': y_hidden,
            'x_block': x_block,
            'y_block': y_block,
            'wall_x': wall_x,
            'wall_y': wall_y,
            'front_view_x': front_view_x,
            'front_view_y': front_view_y,
            'rear_view_x': rear_poly.exterior.xy[0],
            'rear_view_y': rear_poly.exterior.xy[1],
            'passenger_mirror_view_x': passenger_poly.exterior.xy[0],
            'passenger_mirror_view_y': passenger_poly.exterior.xy[1],
            'driver_mirror_view_x': driver_poly.exterior.xy[0],
            'driver_mirror_view_y': driver_poly.exterior.xy[1],
            'center': driver_seat_loc,
            'ego_car': ego_bbox,
            'visibility_data': visibility_data,
        }

    # Desc: This function creates a cone-shape polygon. To do that it use
    # five inputs(point, radius, angle, opening, resolution).
    #   'point': is the vertex of the cone.
    #   'radius': is the longitude from 'point' to any point in the arc.
    #   'angle': is the direcction of the cone.
    #   'resolution': is the number of degrees one point and the next in the arc.
    # Return: The function returns a Polygon object with the shape of
    #   a cone with the above characteristics.
    def _create_cone(self, point, radio, angle, opening, resolution=1):
        # Define the list for the points of the cone-shape polygon
        p = []

        # The fisrt point will be the vertex of the cone
        p.append(vis.Point(point[0], point[1]))

        # Define the start and end of the arc
        start = angle - opening
        end = angle + opening

        for i in range(start, end, resolution):
            # Convert start angle from degrees to radians
            rad = math.radians(i)

            # Calculate the off-set of the first point of the arc
            x = radio * math.cos(rad)
            y = radio * math.sin(rad)

            # Add the off-set to the vertex point
            new_x = point[0] + x
            new_y = point[1] + y

            # Add the first point of the arc to the list
            p.append(vis.Point(new_x, new_y))

        # Add the last point of the arc
        rad = math.radians(end)
        x = radio * math.cos(rad)
        y = radio * math.sin(rad)
        new_x = point[0] + x
        new_y = point[1] + y
        p.append(vis.Point(new_x, new_y))

        return vis.Polygon(p)

    def _save_print(self, polygon):
        end_pos_x = []
        end_pos_y = []

        for i in range(polygon.n()):
            x = polygon[i].x()
            y = polygon[i].y()

            end_pos_x.append(x)
            end_pos_y.append(y)

        return end_pos_x, end_pos_y


class DataError(Exception):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message
