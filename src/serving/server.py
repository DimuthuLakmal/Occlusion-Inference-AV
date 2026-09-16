import argparse
from concurrent import futures

import grpc
import torch
import yaml

from src.dataset import feature_builder
from src.models.v_stsbgat import VSTSBGT
from src.serving.generated import ogm_inference_pb2_grpc
from src.serving.inference_service import OGMInferenceServicer


def create_args():
    parser = argparse.ArgumentParser(description="OGM occupancy prediction gRPC server")
    parser.add_argument('--config', default="../configs/config.yaml", type=str)
    parser.add_argument('--checkpoint', default=None, type=str,
                       help="Path to a model state_dict .pt file. Defaults to config['model']['model_input_path'].")
    parser.add_argument('--port', default=50051, type=int)
    parser.add_argument('--max-workers', default=4, type=int)
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
    semantic_maps = feature_builder.load_semantic_maps(background_images)

    servicer = OGMInferenceServicer(
        model, background_images, semantic_maps, data_config['history_length'], device)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.max_workers))
    ogm_inference_pb2_grpc.add_OGMInferenceServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f'[::]:{args.port}')
    server.start()
    print(f"OGMInferenceService listening on port {args.port} "
         f"(checkpoint={checkpoint_path}, scenes={sorted(background_images.keys())})")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
