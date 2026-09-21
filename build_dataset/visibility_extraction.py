import argparse
import os
import sys

from loguru import logger

from visibility_checker import DataError, VisibilityChecker
from src.utils.tracks_import import read_from_csv


def create_args():
    cs = argparse.ArgumentParser(description="Dataset Tracks Visualizer")
    # --- Input ---
    cs.add_argument('--dataset_dir', default="../data/",
                    help="Path to directory that contains the dataset csv files.", type=str)
    cs.add_argument('--start_recording_id', default=0,
                    help="Starting recording id that visibility data extraction starts from", type=int)
    cs.add_argument('--end_recording_id', default=32,
                    help="Final recording id that visibility data extraction ends from", type=int)

    return vars(cs.parse_args())


def main():
    config = create_args()

    dataset_dir = config["dataset_dir"] + "/"
    start_recording_id = config["start_recording_id"]
    end_recording_id = config["end_recording_id"]

    # You can iterate over multiple recordings here starting from 0 to 32
    for recording in range(start_recording_id, end_recording_id+1):
        recording = "{:02d}".format(int(recording))

        logger.info("Loading recording {}", recording)

        # Create paths to csv files
        tracks_file = dataset_dir + recording + "_tracks.csv"
        tracks_meta_file = dataset_dir + recording + "_tracksMeta.csv"
        recording_meta_file = dataset_dir + recording + "_recordingMeta.csv"
        fixed_blocks_meta_file = dataset_dir + recording + "_fixedBlocks.csv"
        visibility_file = dataset_dir + recording + "_visibilityData.csv"

        # Load csv files
        logger.info("Loading csv files {}, {} and {}", tracks_file, tracks_meta_file, recording_meta_file)
        tracks, static_info, meta_info, fixed_blocks_info = read_from_csv(tracks_file, tracks_meta_file, recording_meta_file,
                                                       fixed_blocks_meta_file, include_px_coordinates=True)

        # Load background image for visualization
        # background_image_path = dataset_dir + recording + "_background.png"
        background_image_path = dataset_dir + "semantic_maps/" + recording + "_background.png"
        if not os.path.exists(background_image_path):
            logger.warning("Background image {} missing. Fallback to using a black background.", background_image_path)
            background_image_path = None
        config["background_image_path"] = background_image_path

        try:
            visibility_checker = VisibilityChecker(config, tracks, static_info, meta_info, fixed_blocks_info, visibility_file)
            visibility_checker.extract_visibility()
        except DataError:
            sys.exit(1)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    main()
