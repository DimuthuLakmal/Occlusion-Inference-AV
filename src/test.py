import argparse

import torch
import yaml

from dataset.data_loader import OGMDataLoader
from models.v_stsbgat import VSTSBGT
from validate import evaluate

import torch.nn as nn


def create_args():
    cs = argparse.ArgumentParser(description="Dataset Tracks Visualizer")
    # --- Input ---
    cs.add_argument('--dataset_dir', default="../data/",
                    help="Path to directory that contains the dataset csv files.", type=str)
    cs.add_argument('--visualizer_params_dir', default="../data/visualizer_params/",
                    help="Name of the recording given by a number with a leading zero.", type=str)
    cs.add_argument('--history_length', default="20",
                    help="Number of previous timesetps that includes in the historical observations of a data entry",
                    type=int)
    cs.add_argument('--dataset', default="ind",
                    help="The dataset to use for training and testing",
                    type=str)
    cs.add_argument('--warmup_batches', default=10,
                    help="Number of initial batches excluded from latency/throughput stats.",
                    type=int)
    cs.add_argument('--compile', action='store_true',
                    help="Compile the model with torch.compile before running inference.")
    cs.add_argument('--compile_mode', default='default',
                    choices=['default', 'reduce-overhead', 'max-autotune'],
                    help="torch.compile mode to use when --compile is set.")
    cs.add_argument('--threshold', default=0.5, type=float,
                    help="Classification threshold to apply to the sigmoid output when "
                         "computing metrics (see find_best_threshold.py to pick one "
                         "tuned on the validation split).")
    # 0.7335

    return vars(cs.parse_args())


if __name__ == '__main__':
    args = create_args()

    with open("../configs/config.yaml", "r") as stream:
        config = yaml.safe_load(stream)
        config['data']['batch_size'] = config['model']['test_batch_size']

    model = VSTSBGT(config['model']).to(config['model']["device"])
    model.load_state_dict(torch.load(config['model']['model_input_path']))  # 144 for full model
    model = model.to(config['model']["device"])

    valid_dataloader = OGMDataLoader(config['data'], phase='test').create_dataloader()

    if args['compile']:
        print(f"Compiling model with torch.compile(mode='{args['compile_mode']}')...")
        model = torch.compile(model, dynamic=True, mode=args['compile_mode'])

    test_loss = evaluate(model, valid_dataloader, config['model']["device"], test=True,
                          warmup_batches=args['warmup_batches'], threshold=args['threshold'])
    print(f'Validation loss {test_loss}')