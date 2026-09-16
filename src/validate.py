import argparse
import logging
import os

import numpy as np

import torch
import torch.nn as nn
import cv2
import time

from src.utils.metrics import accumulate_counts, compute_metrics_from_counts


def percentile(values, p):
    return float(np.percentile(values, p))


def evaluate(model, valid_data_loader, device, writer=None, epoch=0, test=False, warmup_batches=10):
    # Loading background images
    background_images = {}
    for scene_id in [0, 7, 8, 18, 19]:
        # Store background images for scenes
        bg_path = os.path.join("../data/", 'maps', f"{scene_id}_background.png")
        img = cv2.imread(bg_path)
        background_images[int(scene_id)] = img

    model.eval()

    loss_fn_aggregated = nn.BCEWithLogitsLoss(reduction='none')

    using_cuda = torch.device(device).type == "cuda"
    if using_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    latencies_ms = []
    n_timed_samples = 0

    with torch.no_grad():  # Example: 10 epochs

        loss_sum = 0.0
        mask_count_sum = 0.0
        tp_sum = fp_sum = fn_sum = tn_sum = 0.0
        y_true_all = []
        y_prob_all = []
        batch_itr = 0

        edge_dropped_per_head = None  # lazily-sized torch.float64 [H] once H is known
        edge_total = 0.0

        wall_start = time.perf_counter()

        for batch_idx, (inputs, target) in enumerate(valid_data_loader):

            # Move data to the correct device
            targets = target.to(device)
            for k, v in inputs.items():
                if k != 'edge_index' and k != 'edge_weights':
                    inputs[k] = v.to(device)

            mask = inputs['mask'].squeeze()  # Mask indicates the non-padded cells (1: valid, 0: padded)
            # Masking is applied to ignore unwanted cells
            mask_fixed_blocks = (
                    targets != 3).squeeze()  # Cells occupied with fixed blocks are marked with a 3 in the target
            mask = mask * mask_fixed_blocks  # Consider cells occupied with fixed blocks as not padded

            if using_cuda:
                torch.cuda.synchronize(device)
            batch_start = time.perf_counter()

            outputs, gates, l2_loss, z_masks = model(inputs)

            if using_cuda:
                torch.cuda.synchronize(device)
            batch_end = time.perf_counter()

            if batch_idx >= warmup_batches:
                latencies_ms.append((batch_end - batch_start) * 1000.0)
                n_timed_samples += target.shape[0]

            # z_masks: list of [E_i, H] per-edge, per-head L0 gates (one entry per sample -
            # edge count varies per sample). z == 0 means that edge/head was fully gated out
            # (exact, via the hardtanh clamp in sgat_utils.py) - a real dropped edge, not a
            # thresholding guess.
            for zm in z_masks:
                zm = zm.detach()
                if edge_dropped_per_head is None:
                    edge_dropped_per_head = torch.zeros(zm.shape[1], dtype=torch.float64)
                edge_dropped_per_head += (zm == 0).sum(dim=0).cpu().double()
                edge_total += zm.shape[0]

            outputs = outputs.squeeze()
            outputs_sig = nn.Sigmoid()(outputs)

            # Calculate the binary cross-entropy loss
            targets = targets.squeeze() * mask
            outputs = outputs * mask  # Apply mask to outputs
            outputs_sig = outputs_sig * mask  # Apply mask to outputs

            loss_aggregated = loss_fn_aggregated(outputs, targets)
            loss_aggregated = loss_aggregated * mask
            # scene_wise_loss = list(loss_aggregated.detach().cpu().numpy())

            # Accumulate raw sums/counts across all batches so that loss and
            # metrics are computed ONCE over the whole validation set at the
            # end, rather than as an unweighted average of per-batch values -
            # the latter is not invariant to test_batch_size.
            loss_sum += loss_aggregated.sum().item()
            mask_count_sum += mask.sum().item()

            tp, fp, fn, tn, y_true, y_prob = accumulate_counts(outputs_sig, targets, mask)
            tp_sum += tp
            fp_sum += fp
            fn_sum += fn
            tn_sum += tn
            y_true_all.append(y_true)
            y_prob_all.append(y_prob)

            batch_itr += 1

            # scene_data = [(arr[0], scene_loss) for (arr, scene_loss) in zip(list(inputs["scene_id"].detach().cpu().numpy()), scene_wise_loss)]

            # Check sparse edge cases
            # cell_obs = inputs['hidden_ogm_cells']
            # v_obs = inputs['historical_adjacent_obs']
            #
            # preds = (outputs_sig >= 0.5).float()
            #
            # for b_i in range(cell_obs.shape[0]):
            #     map_img = background_images[inputs['scene_id'][b_i].item()]
            #
            #     x_coords = v_obs[b_i, :, :, 0:1] * map_img.shape[1]
            #     y_coords = v_obs[b_i, :, :, 1:2] * map_img.shape[0]
            #
            #     cell_x_coords = (cell_obs[b_i, :, 0:1] * map_img.shape[1]).squeeze()
            #     cell_y_coords = (cell_obs[b_i, :, 1:2] * map_img.shape[0]).squeeze()
            #
            #     if preds[b_i].item() == targets[b_i].item():
            #         z_masks[b_i] = z_masks[b_i].permute(1, 0)
            #         for h, z_mask in enumerate(z_masks[b_i]):
            #             edge_val = z_mask.detach().cpu().numpy()
            #             cell_pts = (int(cell_x_coords.item()), int(cell_y_coords.item()))
            #
            #             map_img_t = map_img.copy()
            #             for j in range(edge_val.shape[0]):
            #                 pt = (int(x_coords[j, -1, 0].item()), int(y_coords[j, -1, 0].item()))
            #                 for t_ext in range(5):
            #                     if pt[0] == 0 and pt[1] == 0:
            #                         pt = (int(x_coords[j, -1 - t_ext, 0].item()), int(y_coords[j, -1 - t_ext, 0].item()))
            #
            #                 if pt[0] == 0 and pt[1] == 0:
            #                     continue
            #
            #                 if edge_val[j] >= 0.1:
            #                     # cv2.line(map_img_t, cell_pts, pt, lerp_color(edge_val[j]), 2)
            #                     cv2.line(map_img_t, cell_pts, pt, (3, 252, 232), 2)
            #                 else:
            #                     draw_dotted_line(map_img_t, cell_pts, pt, (3, 132, 252), 2)
            #
            #
            #                 cv2.circle(map_img_t, pt, 5, (0, 255, 0), -1)
            #
            #             if targets[b_i].item() == 1:
            #                 cv2.circle(map_img_t, cell_pts, 5, (235, 52, 52), -1)
            #             else:
            #                 cv2.circle(map_img_t, cell_pts, 5, (210, 3, 252), -1)
            #
            #             cv2.imwrite(f'../results/edge_masks/{batch_idx}_{b_i}_{h}.png', map_img_t)

        wall_elapsed = time.perf_counter() - wall_start

    valid_loss = loss_sum / max(mask_count_sum, 1)
    m = compute_metrics_from_counts(
        tp_sum, fp_sum, fn_sum, tn_sum,
        y_true=np.concatenate(y_true_all) if y_true_all else None,
        y_prob=np.concatenate(y_prob_all) if y_prob_all else None,
    )

    if edge_total > 0:
        edge_drop_pct_per_head = (edge_dropped_per_head / edge_total * 100).tolist()
        edge_drop_pct_overall = float(
            edge_dropped_per_head.sum() / (edge_total * edge_dropped_per_head.numel()) * 100)
    else:
        edge_drop_pct_per_head, edge_drop_pct_overall = [], None

    latencies_ms = np.array(latencies_ms, dtype=np.float64)
    timed_elapsed_s = latencies_ms.sum() / 1000.0
    profile = {
        "num_batches_total": batch_itr,
        "num_batches_warmup": min(warmup_batches, batch_itr),
        "num_batches_timed": int(latencies_ms.shape[0]),
        "num_samples_timed": n_timed_samples,
        "latency_ms_mean": float(latencies_ms.mean()) if latencies_ms.size else float("nan"),
        "latency_ms_std": float(latencies_ms.std()) if latencies_ms.size else float("nan"),
        "latency_ms_min": float(latencies_ms.min()) if latencies_ms.size else float("nan"),
        "latency_ms_p50": percentile(latencies_ms, 50) if latencies_ms.size else float("nan"),
        "latency_ms_p95": percentile(latencies_ms, 95) if latencies_ms.size else float("nan"),
        "latency_ms_p99": percentile(latencies_ms, 99) if latencies_ms.size else float("nan"),
        "latency_ms_max": float(latencies_ms.max()) if latencies_ms.size else float("nan"),
        "throughput_samples_per_sec": (n_timed_samples / timed_elapsed_s) if timed_elapsed_s > 0 else float("nan"),
        "wall_clock_total_sec": wall_elapsed,
    }
    if using_cuda:
        profile["peak_gpu_memory_allocated_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        profile["peak_gpu_memory_reserved_mb"] = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

    if not test:
        writer.add_scalar("val/loss_epoch", valid_loss, epoch)
        writer.add_scalar("val/accuracy_epoch", m["accuracy"], epoch)
        writer.add_scalar("val/precision_epoch", m["precision"], epoch)
        writer.add_scalar("val/recall_epoch", m["recall"], epoch)
        writer.add_scalar("val/acc_free_epoch", m["acc_free"], epoch)
        writer.add_scalar("val/f1_epoch", m["f1"], epoch)
        if m["roc_auc"] is not None:
            writer.add_scalar("val/roc_auc_epoch", m["roc_auc"], epoch)
        if edge_drop_pct_overall is not None:
            writer.add_scalar("val/edge_drop_pct_overall", edge_drop_pct_overall, epoch)
            for h, pct in enumerate(edge_drop_pct_per_head):
                writer.add_scalar(f"val/edge_drop_pct_head{h}", pct, epoch)

        print(f'Epoch {epoch}, Validation Loss: {valid_loss}')
        logging.info(f'Epoch {epoch}, Validation Loss: {valid_loss}')
    else:
        edge_drop_str = (f'{edge_drop_pct_overall:.2f}%, per-head: '
                         f'{[f"{p:.2f}%" for p in edge_drop_pct_per_head]}'
                         if edge_drop_pct_overall is not None else "N/A")

        print("\n" + "-" * 50)
        print("Task metrics")
        print("-" * 50)
        print(f'Test Loss: {valid_loss}, Accuracy: {m["accuracy"]}, Precision: {m["precision"]}, '
              f'Recall: {m["recall"]}, Accuracy Free: {m["acc_free"]} F1: {m["f1"]}, '
              f'ROC AUC: {m["roc_auc"] if m["roc_auc"] is not None else "N/A"}, '
              f'Edge Drop % (overall): {edge_drop_str}')

        print("\n" + "-" * 50)
        print("Performance profile")
        print("-" * 50)
        print(
            f"batches: {profile['num_batches_total']} (warmup {profile['num_batches_warmup']}, "
            f"timed {profile['num_batches_timed']})"
        )
        print(
            f"latency (ms): mean {profile['latency_ms_mean']:.2f} | std {profile['latency_ms_std']:.2f} | "
            f"P50 {profile['latency_ms_p50']:.2f} | P95 {profile['latency_ms_p95']:.2f} | "
            f"P99 {profile['latency_ms_p99']:.2f} | min {profile['latency_ms_min']:.2f} | "
            f"max {profile['latency_ms_max']:.2f}"
        )
        print(f"throughput: {profile['throughput_samples_per_sec']:.2f} samples/sec")
        print(f"wall clock total: {profile['wall_clock_total_sec']:.2f} s")
        if "peak_gpu_memory_allocated_mb" in profile:
            print(
                f"peak GPU memory: {profile['peak_gpu_memory_allocated_mb']:.1f} MB allocated / "
                f"{profile['peak_gpu_memory_reserved_mb']:.1f} MB reserved"
            )

    return valid_loss

def draw_dotted_line(img, pt1, pt2, color, thickness=1, dot_length=5, gap_length=5):
    """
    Draws a dotted line on an OpenCV image.

    Args:
        img (numpy.ndarray): The image on which to draw the line.
        pt1 (tuple): The starting point (x, y) of the line.
        pt2 (tuple): The ending point (x, y) of the line.
        color (tuple): The color of the line in BGR format (e.g., (255, 0, 0) for blue).
        thickness (int): The thickness of the dots.
        dot_length (int): The length of each individual dot segment.
        gap_length (int): The length of the gap between dots.
    """
    dist = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
    num_segments = int(dist / (dot_length + gap_length))

    for i in range(num_segments):
        start_ratio = i * (dot_length + gap_length) / dist
        end_ratio = (i * (dot_length + gap_length) + dot_length) / dist

        if end_ratio > 1:  # Prevent drawing beyond the line's end
            end_ratio = 1

        # Calculate start and end points for the current dot segment
        dot_start_x = int(pt1[0] * (1 - start_ratio) + pt2[0] * start_ratio)
        dot_start_y = int(pt1[1] * (1 - start_ratio) + pt2[1] * start_ratio)
        dot_end_x = int(pt1[0] * (1 - end_ratio) + pt2[0] * end_ratio)
        dot_end_y = int(pt1[1] * (1 - end_ratio) + pt2[1] * end_ratio)

        cv2.line(img, (dot_start_x, dot_start_y), (dot_end_x, dot_end_y), color, thickness)


def lerp_color(v):
    """
    v ∈ [0,1] and returns a color between Yellow → Orange in BGR format.
    """
    # Yellow RGB: (255, 255, 0)
    # Orange RGB: (255, 165, 0)

    yellow = np.array([255, 255, 0], dtype=np.float32)
    orange = np.array([255, 0, 0], dtype=np.float32)

    rgb = yellow * (1 - v) + orange * v  # Linear interpolation
    bgr = rgb[::-1]  # Convert RGB → BGR for OpenCV

    return tuple(int(c) for c in bgr)
