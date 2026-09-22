import argparse
from concurrent import futures

import cv2
import grpc
import torch
import yaml

from src.dataset import feature_builder
from src.models.v_stsbgat import VSTSBGT
from src.serving.generated import ogm_inference_pb2_grpc
from src.serving.inference_service import OGMInferenceServicer, VISUALIZATION_KINDS


def create_args():
    parser = argparse.ArgumentParser(description="OGM occupancy prediction gRPC server")
    parser.add_argument('--config', default="../configs/config.yaml", type=str)
    parser.add_argument('--checkpoint', default=None, type=str,
                       help="Path to a model state_dict .pt file. Defaults to config['model']['model_input_path'].")
    parser.add_argument('--port', default=50051, type=int)
    parser.add_argument('--max-workers', default=4, type=int)
    parser.add_argument('--threshold', default=0.5, type=float,
                       help="Probability at/above which a cell is labelled is_occupied (see "
                            "find_best_threshold.py to pick one tuned on the validation split).")
    parser.add_argument('--visualizations', nargs='*', choices=VISUALIZATION_KINDS,
                       default=list(VISUALIZATION_KINDS),
                       help="Which debug videos to generate: 'edges' (z-mask edge grid per head) and/or "
                            "'gradcam' (per-scale Grad-CAM heatmaps; costs an extra forward+backward per "
                            "occupied cell). Default: both. Pass the flag with no values "
                            "(--visualizations) to disable all.")
    parser.add_argument('--viz-output-dir', default="../results/serving_visualizations", type=str,
                       help="Directory under which each call's z-mask visualizations are saved, "
                            "in a per-call call_NNNNN subfolder.")
    parser.add_argument('--perturbed-map-image', default=None, type=str,
                       help="Optional path to an alternate background/semantic-map PNG (same "
                            "palette convention as the real background images). When set, each "
                            "GradCAM call additionally re-runs the forward+backward pass with "
                            "this image substituted for map_obs, for the same cells the original "
                            "image found occupied, producing a second attribution + probability "
                            "per cell for vision-backbone ablation. Not scene-aware - it's your "
                            "responsibility to point it at a perturbed version of the scene you're "
                            "actually querying.")
    return parser.parse_args()


def load_model(config, checkpoint_path, device):
    model = VSTSBGT(config['model']).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def serve():
    args = create_args()
    with open(args.config, "r") as stream:
        config = yaml.safe_load(stream)

    device = config['model']['device']
    checkpoint_path = args.checkpoint or config['model']['model_input_path']
    model = load_model(config, checkpoint_path, device)

    data_config = config['data']
    scene_ids = [f"{s:02d}" for s in range(data_config['start_scene'], data_config['end_scene'] + 1)]
    background_images = feature_builder.load_background_images(
        data_config['dataset_dir'], scene_ids, data_config['start_scene'], data_config['end_scene'])
    true_map_images = feature_builder.load_background_images(
        data_config['dataset_dir'], scene_ids, data_config['start_scene'], data_config['end_scene'],
        subdir='')
    semantic_maps = feature_builder.load_semantic_maps(background_images)

    perturbed_map_obs = None
    if args.perturbed_map_image:
        perturbed_img = cv2.imread(args.perturbed_map_image)
        if perturbed_img is None:
            raise FileNotFoundError(f"Could not read perturbed map image: {args.perturbed_map_image}")
        perturbed_map_obs = feature_builder.load_semantic_maps({0: perturbed_img})[0]

    servicer = OGMInferenceServicer(
        model, background_images, semantic_maps, data_config['history_length'], device,
        viz_output_dir=args.viz_output_dir, true_map_images=true_map_images,
        threshold=args.threshold, visualizations=args.visualizations,
        perturbed_map_obs=perturbed_map_obs)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.max_workers))
    ogm_inference_pb2_grpc.add_OGMInferenceServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f'[::]:{args.port}')
    server.start()
    print(f"OGMInferenceService listening on port {args.port} "
         f"(checkpoint={checkpoint_path}, threshold={args.threshold}, "
         f"visualizations={args.visualizations or 'none'}, "
         f"scenes={sorted(background_images.keys())}, "
         f"perturbed_map_image={args.perturbed_map_image or 'none'})")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
