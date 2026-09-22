import os
import threading

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import grpc

from src.dataset import feature_builder
from src.serving.generated import ogm_inference_pb2, ogm_inference_pb2_grpc
from src.validate import draw_dotted_line

EDGE_KEEP_THRESHOLD = 0.1  # matches the sparse-edge visualization convention in validate.py
EDGE_KEEP_COLOR = (3, 252, 232)
EDGE_DROP_COLOR = (3, 132, 252)
VEHICLE_COLOR = (0, 255, 0)
OCCUPIED_CELL_COLOR = (235, 52, 52)
MAX_CELLS_PER_FRAME = 3  # cap rows per grid frame
VISUALIZATION_KINDS = ("edges", "gradcam")  # selectable debug videos, see OGMInferenceServicer.__init__
HEADER_HEIGHT = 55
PYRAMID_STAGES = ("s4", "s8", "s16", "s32")  # shared by ConvNeXtMapEncoder and RegNetY800MFMapEncoder


class OGMInferenceServicer(ogm_inference_pb2_grpc.OGMInferenceServiceServicer):
    """
    Builds the same tensors OGMDataset would build for a frame (via feature_builder,
    shared with training) from a live PredictOccupancyRequest, and runs the model on
    all of the request's occluded cells in a single forward pass.
    """

    def __init__(self, model, background_images, semantic_maps, history_length, device,
                viz_output_dir="../results/serving_visualizations", true_map_images=None,
                threshold=0.5, visualizations=VISUALIZATION_KINDS, perturbed_map_obs=None):
        self.model = model
        # Optional [K,H,W] one-hot semantic map (see server.py --perturbed-map-image) used for
        # vision-backbone ablation: GradCAM re-runs forward+backward with this substituted for
        # map_obs, for whichever cells the *original* image already found occupied, so the two
        # explanations can be compared. None disables the extra pass entirely.
        self.perturbed_map_obs = perturbed_map_obs
        self.threshold = threshold  # sigmoid probability at/above which a cell is labelled occupied
        unknown = set(visualizations) - set(VISUALIZATION_KINDS)
        if unknown:
            raise ValueError(f"Unknown visualization(s) {sorted(unknown)}; choose from {VISUALIZATION_KINDS}")
        # Which debug videos to produce: "edges" (z-mask grid) and/or "gradcam". Empty = none,
        # in which case no output dir, video streams or Grad-CAM hook are set up at all.
        self.visualizations = frozenset(visualizations)
        self.background_images = background_images
        self.semantic_maps = semantic_maps
        self.history_length = history_length
        self.device = device
        # Used only as the visualization canvas (real aerial photo); background_images (the
        # color-by-class semantic map) remains what coordinate-normalization math is based on,
        # and is the fallback here if a scene has no true-map image loaded.
        self.true_map_images = true_map_images or {}

        # Two running videos for the server's lifetime, one frame per call that has at least one
        # occupied cell (calls with none are skipped, contributing no frame): "heads" (the z-mask
        # per-head grid) and "gradcam" (per-scale Grad-CAM heatmaps, see _build_gradcam_grid). A
        # GIF's per-frame duration is unreliably honored by many viewers (plays back much faster
        # than set), so these are actual mp4s with an explicit fps instead. Frames are kept in
        # memory and the whole file is rewritten after every call - that makes the file appear
        # and stay current immediately while the server runs, at the cost of O(frames-so-far)
        # work per call; fine for a debug/analysis tool.
        self._video_fps = 0.5  # 1 frame every 2s - adjust to taste
        self._video_streams = {}
        if "edges" in self.visualizations:
            self._video_streams["heads"] = {
                "frames": [], "path": os.path.join(viz_output_dir, "simulation.mp4"), "frame_size": None}
        if "gradcam" in self.visualizations:
            self._video_streams["gradcam"] = {
                "frames": [], "path": os.path.join(viz_output_dir, "simulation_gradcam.mp4"), "frame_size": None}
        if self._video_streams:
            os.makedirs(viz_output_dir, exist_ok=True)
        self._call_counter = 0
        self._lock = threading.Lock()

        # Per-scale Grad-CAM: capture the raw ConvNeXt/RegNet pyramid via a forward hook (rather
        # than threading it through VSTSBGT.forward's return value, which every caller - train.py,
        # validate.py, test.py, find_best_threshold.py, this file's own no-grad forward above -
        # unpacks as a fixed-size tuple). retain_grad() is needed because the pyramid tensors are
        # non-leaf intermediates, which don't populate .grad after backward() otherwise.
        self._last_pyramid = None
        if "gradcam" in self.visualizations:
            self.model.semantic_context_encoder.register_forward_hook(self._capture_pyramid_hook)
        self._gradcam_lock = threading.Lock()  # serializes the grad-enabled forward+backward section

    def _capture_pyramid_hook(self, module, inputs, output):
        pyramid = output["map_pyramid"]
        for t in pyramid.values():
            if t.requires_grad:  # this hook also fires on the earlier torch.no_grad() forward
                t.retain_grad()
        self._last_pyramid = pyramid

    def PredictOccupancy(self, request, context):
        background_img = self.background_images.get(request.scene_id)
        if background_img is None:
            context.abort(grpc.StatusCode.NOT_FOUND, f"Unknown scene_id: {request.scene_id}")

        if len(request.vehicles) == 0:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "At least one vehicle observation is required")
        if len(request.cells) == 0:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "At least one occluded cell is required")

        expected_t = self.history_length + 1
        for vehicle in request.vehicles:
            if len(vehicle.timesteps) != expected_t:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"vehicle '{vehicle.track_id}' has {len(vehicle.timesteps)} timesteps, "
                    f"expected exactly {expected_t}")

        historical_adjacent_obs = {
            idx: self._build_raw_vehicle_obs(vehicle, expected_t)
            for idx, vehicle in enumerate(request.vehicles)
        }
        last_recorded_t = feature_builder.compute_last_recorded_t(historical_adjacent_obs)

        cells_xy_norm = [
            [cell.cx / (background_img.shape[1] - 1), cell.cy / (background_img.shape[0] - 1)]
            for cell in request.cells
        ]

        edge_weights, edge_index = feature_builder.extract_edge_info(
            historical_adjacent_obs, cells_xy_norm, last_recorded_t)
        historical_adjacent_input, seq_mask = feature_builder.build_vehicle_tensor(
            historical_adjacent_obs, request.scene_id)
        map_obs = self.semantic_maps[request.scene_id]

        inputs = self._to_batch_of_one(
            historical_adjacent_input, seq_mask, cells_xy_norm, map_obs, edge_weights, edge_index)

        self.model.eval()
        with torch.no_grad():
            out_fc, _gates, _l2_loss, _z_mask = self.model(inputs)
        probs = torch.sigmoid(out_fc).squeeze(-1).squeeze(0).cpu().numpy()  # [N_cells]
        z_mask = _z_mask[0]  # batch-of-one: single bipartite graph per call

        response = ogm_inference_pb2.PredictOccupancyResponse()
        occupied_indices = []
        for i, (cell, prob) in enumerate(zip(request.cells, probs)):
            is_occupied = bool(prob >= self.threshold)
            response.predictions.add(
                cx=cell.cx, cy=cell.cy,
                occupancy_probability=float(prob),
                is_occupied=is_occupied)
            if is_occupied:
                occupied_indices.append(i)

        if occupied_indices and self.visualizations:
            with self._lock:
                self._call_counter += 1
                call_id = self._call_counter

            canvas = self.true_map_images.get(request.scene_id, background_img)
            capped_indices = occupied_indices[:MAX_CELLS_PER_FRAME]

            if "edges" in self.visualizations:
                grid = self._build_head_mask_grid(capped_indices, request.cells, canvas,
                                                  historical_adjacent_obs, last_recorded_t,
                                                  z_mask, edge_index)
                self._append_frame(grid, "heads", call_id)

            if "gradcam" in self.visualizations:
                gradcam_grid = self._build_gradcam_grid(capped_indices, canvas, inputs)
                self._append_frame(gradcam_grid, "gradcam", call_id)

        return response

    def _append_frame(self, frame_bgr, stream_key, call_id):
        """Appends one BGR frame to the named running video (see __init__), rewriting the whole
        file so it stays immediately viewable. Thread-safe."""
        stream = self._video_streams[stream_key]
        with self._lock:
            # Burn in the call number as an on-screen label identifying which call each frame
            # corresponds to - shared across streams so "call N" lines up between them.
            cv2.putText(frame_bgr, f"call {call_id}", (8, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Video requires every frame in the file to share one size; the grid's size varies
            # with how many cells were occupied that call, so later frames are resized to
            # match the first one.
            if stream["frame_size"] is None:
                stream["frame_size"] = (frame_rgb.shape[1], frame_rgb.shape[0])
            elif (frame_rgb.shape[1], frame_rgb.shape[0]) != stream["frame_size"]:
                frame_rgb = cv2.resize(frame_rgb, stream["frame_size"])

            stream["frames"].append(frame_rgb)
            # H.264 requires even (and preferably 16-block-aligned) dimensions; our grid sizes
            # are neither in general, so ffmpeg pads them slightly (its default behavior) rather
            # than passing macro_block_size=1, which produced a corrupt file when a dimension
            # was odd (H.264 flatly rejects odd width/height).
            imageio.mimsave(stream["path"], stream["frames"], fps=self._video_fps)

    @staticmethod
    def _build_head_mask_grid(occupied_indices, cells, canvas, historical_adjacent_obs,
                              last_recorded_t, z_mask, edge_index):
        """Builds one grid image for this call: one row per occupied cell, one column per GAT
        head. Each cell of the grid is that occupied cell's z-mask-gated connections to every
        adjacent vehicle for that one head, drawn over `canvas` (the true map image) - the same
        per-(cell,head) drawing convention as the debug code in validate.py, just tiled instead
        of saved separately."""
        edge_src, edge_dst = edge_index
        width, height = canvas.shape[1], canvas.shape[0]
        num_heads = z_mask.shape[1]

        rows = []
        for cell_idx in occupied_indices:
            cell = cells[cell_idx]
            edges_for_cell = [k for k, dst in enumerate(edge_dst) if dst == cell_idx]
            cell_pt = (int(cell.cx), int(cell.cy))

            sub_images = []
            for h in range(num_heads):
                img = canvas.copy()

                for k in edges_for_cell:
                    vehicle_idx = edge_src[k]
                    t = last_recorded_t[vehicle_idx]
                    if t is None:
                        continue  # vehicle was never actually recorded

                    x_norm, y_norm = historical_adjacent_obs[vehicle_idx][t][:2]
                    vehicle_pt = (int(x_norm * (width - 1)), int(y_norm * (height - 1)))

                    if z_mask[k, h].item() >= EDGE_KEEP_THRESHOLD:
                        cv2.line(img, cell_pt, vehicle_pt, EDGE_KEEP_COLOR, 2)
                    else:
                        draw_dotted_line(img, cell_pt, vehicle_pt, EDGE_DROP_COLOR, 2)

                    cv2.circle(img, vehicle_pt, 5, VEHICLE_COLOR, -1)

                cv2.circle(img, cell_pt, 5, OCCUPIED_CELL_COLOR, -1)
                sub_images.append(img)

            rows.append(sub_images)

        return OGMInferenceServicer._stack_labeled_grid(
            rows, [f"Head {h + 1}" for h in range(num_heads)])

    @staticmethod
    def _stack_labeled_grid(rows, column_labels):
        """Tiles `rows` (a list of rows, each a list of same-sized BGR images, one per column)
        into a single grid image with a labeled header strip on top - shared by
        _build_head_mask_grid and _build_gradcam_grid."""
        column_width = rows[0][0].shape[1]
        grid_body = cv2.vconcat([cv2.hconcat(row) for row in rows])

        header = np.zeros((HEADER_HEIGHT, grid_body.shape[1], 3), dtype=np.uint8)
        for col, label in enumerate(column_labels):
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
            x = col * column_width + (column_width - text_w) // 2
            y = (HEADER_HEIGHT + text_h) // 2
            cv2.putText(header, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                       (255, 255, 255), 3, cv2.LINE_AA)

        return cv2.vconcat([header, grid_body])

    def _compute_attribution_rows(self, out_fc, features, stages, occupied_indices, canvas,
                                  inputs, row_tag):
        """Per-cell, per-scale Gradient × Activation for one forward pass's pyramid.

        Red:   positive signed attribution.
        Blue:  negative signed attribution.
        Clear: attribution near zero.

        Each panel is normalized independently for display. `row_tag` (e.g. "orig" /
        "perturbed") is burned onto each row's first panel so multiple passes over the same
        cells stay distinguishable once stacked into one grid. Returns (rows, probs), where
        probs is {cell_idx: sigmoid(logit)} under this pass.
        """
        rows = []
        probs = {}

        for row_idx, cell_idx in enumerate(occupied_indices):
            # Explain this cell's occupied-class logit.
            target = out_fc[0, cell_idx, 0]
            probs[cell_idx] = torch.sigmoid(target).detach().item()

            # Use returned gradients directly instead of accumulated .grad.
            gradients = torch.autograd.grad(
                outputs=target,
                inputs=features,
                retain_graph=(
                        row_idx < len(occupied_indices) - 1
                ),
                create_graph=False,
                allow_unused=True,
            )

            row = []

            for stage, feature, gradient in zip(
                    stages, features, gradients
            ):
                if gradient is None:
                    raise RuntimeError(
                        f"{stage} is disconnected from the target logit."
                    )

                activation = feature[0]  # [C, H, W]
                grad = gradient[0]  # [C, H, W]

                # Spatially local product; no gradient pooling or ReLU.
                signed = (activation * grad).sum(dim=0)

                if not torch.isfinite(signed).all().item():
                    raise RuntimeError(
                        f"Non-finite attribution at {stage}, "
                        f"cell {cell_idx}."
                    )

                # Inspect raw magnitudes before display normalization.
                print(
                    f"cell={cell_idx}, tag={row_tag}, stage={stage}, "
                    f"logit={target.detach().item():.4f}, "
                    f"grad_max={grad.detach().abs().max().item():.3e}, "
                    f"attr_min={signed.detach().min().item():.3e}, "
                    f"attr_max={signed.detach().max().item():.3e}"
                )

                signed_np = signed.detach().float().cpu().numpy()

                resized = cv2.resize(
                    signed_np,
                    (canvas.shape[1], canvas.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

                # Symmetric display scale around zero.
                peak = float(np.abs(resized).max())
                if peak > 0.0:
                    normalized = resized / peak
                else:
                    normalized = np.zeros_like(resized)

                positive = np.clip(normalized, 0.0, 1.0)
                negative = np.clip(-normalized, 0.0, 1.0)

                # OpenCV uses BGR: positive=red, negative=blue.
                colors = np.zeros_like(canvas, dtype=np.float32)
                colors[..., 2] = np.where(positive > 0, 255.0, 0.0)
                colors[..., 0] = np.where(negative > 0, 255.0, 0.0)

                # Zero attribution leaves the original canvas unchanged.
                opacity = (
                        0.65 * np.abs(normalized)
                )[..., None]

                overlay = (
                        canvas.astype(np.float32) * (1.0 - opacity)
                        + colors * opacity
                )
                overlay = np.clip(overlay, 0, 255).astype(np.uint8)

                # Mark the target cell using its normalized coordinates.
                xy = inputs["hidden_ogm_cells"][0, cell_idx]
                cx = int(round(
                    xy[0].detach().item() * (canvas.shape[1] - 1)
                ))
                cy = int(round(
                    xy[1].detach().item() * (canvas.shape[0] - 1)
                ))
                cv2.circle(
                    overlay, (cx, cy), 6,
                    (255, 255, 255), 2,
                )

                row.append(overlay)

            # Label the leftmost panel with the cell/tag/probability so orig vs. perturbed
            # rows for the same cell are identifiable once stacked into one grid.
            label = f"cell {cell_idx} {row_tag} p={probs[cell_idx]:.2f}"
            cv2.putText(row[0], label, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                       (255, 255, 255), 2, cv2.LINE_AA)

            rows.append(row)

        return rows, probs

    def _run_gradcam_pass(self, inputs, map_obs_override=None):
        """Runs one grad-enabled forward pass (optionally with map_obs replaced), returning
        (out_fc, features, stages). Must be called inside self._gradcam_lock /
        torch.enable_grad(); resets self._last_pyramid in a finally block."""
        pass_inputs = dict(inputs)
        if map_obs_override is not None:
            pass_inputs["map_obs"] = map_obs_override
        else:
            # Also allows attribution when backbone parameters are frozen.
            pass_inputs["map_obs"] = inputs["map_obs"].detach().clone().requires_grad_(True)

        self.model.eval()
        try:
            # Existing forward hook captures and retains the pyramid.
            out_fc, _, _, _ = self.model(pass_inputs)
            pyramid = self._last_pyramid

            if pyramid is None:
                raise RuntimeError("The pyramid hook did not capture features.")

            stages = list(PYRAMID_STAGES)
            features = tuple(pyramid[stage] for stage in stages)
            return out_fc, features, stages
        finally:
            self._last_pyramid = None

    def _build_gradcam_grid(self, occupied_indices, canvas, inputs):
        """Builds the GradCAM comparison grid for this call's occupied cells (see
        _compute_attribution_rows for the attribution math). When self.perturbed_map_obs is
        set, each cell gets a second row explaining the same cell index under the perturbed
        background image, for vision-backbone ablation - "occupied" is always decided by the
        original image (occupied_indices, computed upstream in PredictOccupancy), never the
        perturbed one."""
        with self._gradcam_lock, torch.enable_grad():
            out_fc, features, stages = self._run_gradcam_pass(inputs)
            orig_rows, orig_probs = self._compute_attribution_rows(
                out_fc, features, stages, occupied_indices, canvas, inputs, row_tag="orig")

            if self.perturbed_map_obs is None:
                return self._stack_labeled_grid(orig_rows, stages)

            perturbed_map_obs = torch.as_tensor(
                self.perturbed_map_obs, dtype=torch.float32, device=self.device,
            ).unsqueeze(0).requires_grad_(True)  # batch is always 1, see _to_batch_of_one
            out_fc_p, features_p, _ = self._run_gradcam_pass(inputs, map_obs_override=perturbed_map_obs)
            pert_rows, pert_probs = self._compute_attribution_rows(
                out_fc_p, features_p, stages, occupied_indices, canvas, inputs, row_tag="perturbed")

            for cell_idx in occupied_indices:
                print(f"cell={cell_idx}: orig_prob={orig_probs[cell_idx]:.4f}, "
                     f"perturbed_prob={pert_probs[cell_idx]:.4f}, "
                     f"delta={pert_probs[cell_idx] - orig_probs[cell_idx]:+.4f}")

            rows = []
            for orig_row, pert_row in zip(orig_rows, pert_rows):
                rows.append(orig_row)
                rows.append(pert_row)

            return self._stack_labeled_grid(rows, stages)

    @staticmethod
    def _build_raw_vehicle_obs(vehicle, expected_t):
        """
        Builds the raw [T, 10] observation array feature_builder expects, matching the
        training-time pickle layout. Column 8 (time_offset = t / history_length, oldest=0,
        current=1) is derived from the timestep's position; column 9 (distance-to-ego) is
        never read by the model, left at 0. A timestep left entirely at its proto default
        (every field == 0) is treated as not-yet-visible/padding, matching training's
        all-zero-row convention.
        """
        history_length = expected_t - 1
        obs = np.zeros((expected_t, 10), dtype=np.float32)
        for t, ts in enumerate(vehicle.timesteps):
            if ts.x_norm == 0 and ts.y_norm == 0 and ts.heading_deg == 0 and \
               ts.x_velocity == 0 and ts.y_velocity == 0 and \
               ts.x_acceleration == 0 and ts.y_acceleration == 0 and \
               ts.vehicle_type == ogm_inference_pb2.CAR:
                continue  # leave this timestep as the zero row already in `obs`

            obs[t] = [ts.x_norm, ts.y_norm, ts.heading_deg,
                     ts.x_velocity, ts.y_velocity,
                     ts.x_acceleration, ts.y_acceleration,
                     float(ts.vehicle_type),
                     t / history_length,
                     0.0]
        return obs

    def _to_batch_of_one(self, historical_adjacent_input, seq_mask, cells_xy_norm, map_obs,
                         edge_weights, edge_index):
        num_vehicles = historical_adjacent_input.shape[0]
        to_tensor = lambda arr, dtype: torch.as_tensor(arr, dtype=dtype, device=self.device).unsqueeze(0)

        return {
            "historical_adjacent_obs": to_tensor(historical_adjacent_input, torch.float32),
            "seq_mask": to_tensor(seq_mask, torch.bool),
            "vehicle_mask": torch.zeros(1, num_vehicles, dtype=torch.bool, device=self.device),
            "hidden_ogm_cells": to_tensor(np.array(cells_xy_norm, dtype=np.float32), torch.float32),
            "map_obs": to_tensor(map_obs.astype(np.float32), torch.float32),
            "edge_weights": [torch.as_tensor(np.expand_dims(np.array(edge_weights, dtype=np.float32), -1),
                                            dtype=torch.float32, device=self.device)],
            "edge_index": [torch.as_tensor(np.array(edge_index, dtype=np.int64),
                                          dtype=torch.int64, device=self.device)],
        }
