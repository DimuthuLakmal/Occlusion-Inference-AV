from collections import OrderedDict
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch


SEMANTIC_PALETTE_BGR = {
    'fixed_blocks': (0, 255, 0),
    'walking_path': (255, 0, 0),
    'vegetation': (255, 0, 150),
    'road': (0, 0, 255),
    'lines': (255, 255, 255),
    'pedestrian_crossing': (255, 150, 0),
    'parking': (0, 255, 255),
    'refuge_island': (90, 90, 90),
    'background': (0, 0, 0)
}


def discover_semantic_colors(
    image_path: str,
    top_k: int = 30,
):
    """
    Inspect the exact colours contained in one of the semantic-map PNGs.

    Important:
        cv2.imread() returns BGR, not RGB.

    Example
    -------
    discover_semantic_colors(
        "semantic_maps/0_background.png"
    )
    """

    image = cv2.imread(
        image_path,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise FileNotFoundError(
            f"Unable to read semantic map: {image_path}"
        )

    pixels = image.reshape(-1, 3)

    colors, counts = np.unique(
        pixels,
        axis=0,
        return_counts=True,
    )

    order = np.argsort(counts)[::-1]

    print(
        f"Found {len(colors)} unique BGR colours."
    )

    for idx in order[:top_k]:
        b, g, r = colors[idx]

        print(
            f"BGR=({b:3d}, {g:3d}, {r:3d}) "
            f"pixels={counts[idx]}"
        )


def semantic_map_to_class_ids(
    image_bgr: np.ndarray,
    palette_bgr: Dict[str, Tuple[int, int, int]],
    max_distance: float = None,
):
    """
    Convert a colour semantic map to class IDs using nearest-colour matching.

    This handles anti-aliased/intermediate colours around semantic boundaries.

    Parameters
    ----------
    image_bgr:
        [H, W, 3], uint8 semantic map loaded by cv2.

    palette_bgr:
        class_name -> (B, G, R)

    max_distance:
        Optional maximum allowed Euclidean colour distance.
        Pixels farther than this from every known class raise an error.

    Returns
    -------
    class_ids:
        [H, W], int64

    class_names:
        list[str]

    min_distances:
        [H, W], distance of each pixel to its assigned palette colour.
        Useful for debugging.
    """

    if image_bgr.ndim != 3 or image_bgr.shape[-1] != 3:
        raise ValueError(
            "Expected image_bgr with shape [H, W, 3]."
        )

    class_names = list(palette_bgr.keys())

    palette = np.asarray(
        [palette_bgr[name] for name in class_names],
        dtype=np.float32,
    )
    # [K, 3]

    pixels = image_bgr.astype(np.float32)

    # [H, W, 1, 3] - [1, 1, K, 3]
    difference = (
        pixels[:, :, None, :]
        - palette[None, None, :, :]
    )

    # Squared Euclidean distance in BGR space.
    distances_sq = np.sum(
        difference ** 2,
        axis=-1,
    )
    # [H, W, K]

    class_ids = np.argmin(
        distances_sq,
        axis=-1,
    ).astype(np.int64)

    min_distances = np.sqrt(
        np.min(
            distances_sq,
            axis=-1,
        )
    )

    if max_distance is not None:

        invalid_mask = (
            min_distances > max_distance
        )

        if invalid_mask.any():
            bad_pixels = image_bgr[
                invalid_mask
            ]

            bad_colours = np.unique(
                bad_pixels,
                axis=0,
            )

            raise ValueError(
                f"{invalid_mask.sum()} pixels are farther "
                f"than {max_distance} from every semantic "
                f"class colour.\n"
                f"Examples: {bad_colours[:20]}"
            )

    return (
        class_ids,
        class_names,
        min_distances,
    )


def resize_class_ids(
    class_ids: np.ndarray,
    output_size: Tuple[int, int],
) -> np.ndarray:
    """
    Resize categorical class IDs.

    Uses nearest-neighbour interpolation intentionally.

    Parameters
    ----------
    output_size:
        (height, width)
    """

    out_h, out_w = output_size

    # Semantic classes are categorical. Do NOT use INTER_AREA,
    # INTER_LINEAR, etc.
    resized = cv2.resize(
        class_ids.astype(np.uint8),
        (out_w, out_h),
        interpolation=cv2.INTER_NEAREST,
    )

    return resized.astype(np.int64)


def class_ids_to_one_hot(
    class_ids: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """
    [H, W] -> [K, H, W]
    """

    one_hot = np.eye(
        num_classes,
        dtype=np.float32,
    )[class_ids]

    # [H, W, K] -> [K, H, W]
    one_hot = np.transpose(
        one_hot,
        (2, 0, 1),
    )

    return one_hot


def semantic_map_to_one_hot(
    image_bgr: np.ndarray,
    palette_bgr: Dict[str, Tuple[int, int, int]],
    output_size: Tuple[int, int] = (224, 224),
):
    class_ids, class_names, distances = semantic_map_to_class_ids(
        image_bgr=image_bgr,
        palette_bgr=palette_bgr,
    )

    class_ids = resize_class_ids(
        class_ids,
        output_size,
    )

    one_hot = class_ids_to_one_hot(
        class_ids,
        num_classes=len(class_names),
    )

    return one_hot, class_ids, class_names