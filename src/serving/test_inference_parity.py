"""
Compares OGMInferenceService (gRPC) predictions against directly running the trained
model on the same test-set annotations and observations -- the same forward-pass +
sigmoid computation src/validate.py's evaluate() performs per-sample.

For each frame in the test annotation set (data/annotations/test/), this reads ALL of
its candidate hidden cells (not just the single cell training randomly narrows to) and
its adjacent-vehicle observation history, sends them to a running OGMInferenceService,
and compares the returned probabilities against calling the model directly.

Requires a running server, e.g. (from src/, with the repo root also on PYTHONPATH -- see
README for why both are needed):
    python -m serving.server --config ../configs/config.yaml --checkpoint ../results/checkpoints/best.pt

Usage (from src/):
    python -m serving.test_inference_parity --checkpoint ../results/checkpoints/best.pt --limit 20
"""
import argparse
import sys

import grpc
import numpy as np
import torch
import yaml

from src.dataset.dataset import OGMDataset
from src.dataset.collate_fn import custom_collate
from src.models.v_stsbgat import VSTSBGT
from src.serving.generated import ogm_inference_pb2, ogm_inference_pb2_grpc

VEHICLE_TYPE_MAP = {0: ogm_inference_pb2.CAR, 1: ogm_inference_pb2.TRUCK_BUS,
                    2: ogm_inference_pb2.BICYCLE, 3: ogm_inference_pb2.PEDESTRIAN}


def build_request(ds, key):
    """
    Builds a PredictOccupancyRequest from a test-set frame's raw annotation cells
    (ds.label_dict[key], ALL candidates, not the single training-narrowed one) and its
    observed adjacent-vehicle history (ds.data_dict[key]) -- the same raw data
    OGMDataset.get_all_candidate_cells_sample reads.
    """
    scene_id = int(key.split("_")[0])
    bg_img = ds.background_images[scene_id]
    ego_heading = ds.data_dict[key]["historical_ego_obs"][-1][2]
    ogm_cells, _ = ds.label_dict[key]
    raw_hist = ds.data_dict[key]["historical_adjacent_obs"]

    req = ogm_inference_pb2.PredictOccupancyRequest(scene_id=scene_id, ego_heading_deg=float(ego_heading))
    for cell in ogm_cells:
        req.cells.add(cx=float(cell[0] * (bg_img.shape[1] - 1)), cy=float(cell[1] * (bg_img.shape[0] - 1)))

    for track_id, obs in raw_hist.items():
        vehicle = req.vehicles.add(track_id=str(track_id))
        for row in obs:
            ts = vehicle.timesteps.add()
            if not np.any(row != 0):
                continue  # padding/not-yet-visible timestep: leave at proto default

            ts.x_norm, ts.y_norm, ts.heading_deg = float(row[0]), float(row[1]), float(row[2])
            ts.x_velocity, ts.y_velocity = float(row[3]), float(row[4])
            ts.x_acceleration, ts.y_acceleration = float(row[5]), float(row[6])
            ts.vehicle_type = VEHICLE_TYPE_MAP.get(int(row[7]), ogm_inference_pb2.CAR)

    return req


def reference_predictions(model, ds, key, device):
    """
    The reference to compare against: exactly what validate.py's evaluate() computes per
    sample (model forward + sigmoid), but over ALL of this frame's candidate cells via
    get_all_candidate_cells_sample instead of the single training-narrowed cell.
    """
    sample = ds.get_all_candidate_cells_sample(key)
    inputs, _ = custom_collate([sample])
    for k, v in inputs.items():
        if k not in ("edge_index", "edge_weights"):
            inputs[k] = v.to(device)

    with torch.no_grad():
        out_fc, *_ = model(inputs)
    return torch.sigmoid(out_fc).squeeze(-1).squeeze(0).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Compare OGMInferenceService predictions against direct model inference on the test set")
    parser.add_argument('--config', default="../configs/config.yaml")
    parser.add_argument('--checkpoint', default="../results/checkpoints/best.pt")
    parser.add_argument('--endpoint', default="localhost:50051")
    parser.add_argument('--limit', type=int, default=20, help="Max number of test-set frames to check (0 = all)")
    parser.add_argument('--atol', type=float, default=1e-3,
                        help="Absolute probability tolerance. The service and this script run the "
                             "model in separate processes, so tiny CPU-BLAS-threading-dependent "
                             "floating point differences in the ResNet/Transformer forward pass are "
                             "expected (~1e-4); this is not a sign of mismatched preprocessing.")
    args = parser.parse_args()

    with open(args.config, "r") as stream:
        config = yaml.safe_load(stream)

    device = "cpu"
    ds = OGMDataset(config['data'], phase='test')

    model = VSTSBGT(config['model']).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    channel = grpc.insecure_channel(args.endpoint)
    stub = ogm_inference_pb2_grpc.OGMInferenceServiceStub(channel)

    keys = ds.keys[:args.limit] if args.limit else ds.keys
    total_cells = 0
    max_abs_diff = 0.0
    failures = []

    for key in keys:
        ref_probs = reference_predictions(model, ds, key, device)
        request = build_request(ds, key)
        try:
            response = stub.PredictOccupancy(request, timeout=30)
        except grpc.RpcError as e:
            failures.append((key, f"gRPC error: {e.code()} {e.details()}"))
            continue

        service_probs = np.array([p.occupancy_probability for p in response.predictions])
        if service_probs.shape != ref_probs.shape:
            failures.append((key, f"shape mismatch: service={service_probs.shape} reference={ref_probs.shape}"))
            continue

        diff = np.abs(service_probs - ref_probs)
        max_abs_diff = max(max_abs_diff, float(diff.max()) if diff.size else 0.0)
        total_cells += len(ref_probs)
        if not np.allclose(service_probs, ref_probs, atol=args.atol):
            failures.append((key, f"max diff {diff.max():.6f} exceeds atol {args.atol} "
                                  f"(service={service_probs}, reference={ref_probs})"))

    print(f"Compared {len(keys)} test-set frames, {total_cells} cells total.")
    print(f"Max abs probability diff: {max_abs_diff:.6f}")
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for key, msg in failures:
            print(f"  {key}: {msg}")
        sys.exit(1)

    print("PASS: service predictions match direct model inference for every test-set frame checked.")


if __name__ == '__main__':
    main()
