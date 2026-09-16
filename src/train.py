import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.data_loader import OGMDataLoader
from models.v_stsbgat import VSTSBGT
import torch.nn as nn

from src.utils.metrics import compute_metrics, accumulate_counts, compute_metrics_from_counts
from validate import evaluate


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

    return vars(cs.parse_args())


def train(model, train_data_loader, valid_data_loader, config):
    logging.basicConfig(
        filename=config['model']['log_file'].format(str(datetime.now())),  # Specify the log file name
        level=logging.INFO,  # Set the logging level (e.g., INFO, DEBUG, WARNING, ERROR, CRITICAL)
        format='%(asctime)s - %(levelname)s - %(message)s',  # Define the log message format
        filemode='a'  # Set the file mode to 'a' for append, or 'w' for overwrite
    )

    # --- TensorBoard ---
    log_dir = config['model']['tb_log_dir']
    writer = SummaryWriter(log_dir=log_dir)
    global_step = 0

    save_ckpt = config['model'].get('save_checkpoints', True)
    save_best_ckpt = config['model'].get('save_best_checkpoint', True)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['model']['lr'])
    optimizer.zero_grad()
    loss_fn_aggregated = nn.BCEWithLogitsLoss(reduction='none')

    best_loss = float('inf')

    if config['model']['use_lr_scheduler']:
        # lambda1 = lambda epoch: 0.9 ** epoch
        # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.985)

    for epoch in range(config['model']['train_epochs']):  # Example: 10 epochs
        model.train()

        loss_sum = 0.0
        mask_count_sum = 0.0
        tp_sum = fp_sum = fn_sum = tn_sum = 0.0
        y_true_all = []
        y_prob_all = []
        batch_itr = 0

        for batch_idx, (inputs, target) in enumerate(train_data_loader):

            # Move data to the correct device
            targets = target.to(config['model']["device"])
            for k, v in inputs.items():
                if k != 'edge_index' and k != 'edge_weights':
                    inputs[k] = v.to(config['model']["device"])

            mask = inputs['mask'].squeeze()  # Mask indicates the non-padded cells (1: valid, 0: padded)
            # Masking is applied to ignore unwanted cells
            mask_fixed_blocks = (
                        targets != 3).squeeze()  # Cells occupied with fixed blocks are marked with a 3 in the target
            mask = mask * mask_fixed_blocks  # Consider cells occupied with fixed blocks as not padded

            outputs, _, l2_loss, z_mask = model(inputs)
            outputs = outputs.squeeze()
            outputs_sig = nn.Sigmoid()(outputs)

            # Calculate the binary cross-entropy loss
            targets = targets.squeeze() * mask
            outputs = outputs * mask  # Apply mask to outputs
            outputs_sig = outputs_sig * mask  # Apply mask to outputs

            loss_aggregated = loss_fn_aggregated(outputs, targets)
            loss_aggregated = loss_aggregated * mask
            loss_avg = loss_aggregated.sum() / (mask.sum().clamp_min(1))

            # --- metrics per batch (masked) ---
            with torch.no_grad():
                # Batch-level metric, only for the per-step TensorBoard/log lines below.
                m = compute_metrics(outputs_sig.detach(), targets.detach(), mask.detach())
                if m["roc_auc"] is not None:
                    # Log batch ROC when available
                    writer.add_scalar("train/roc_auc_batch", m["roc_auc"], global_step)

                # Accumulate raw counts across the whole epoch so epoch-level
                # metrics are computed ONCE at the end (micro-average), rather
                # than as an unweighted average of per-batch ratios - the
                # latter is not invariant to batch_size.
                tp, fp, fn, tn, y_true, y_prob = accumulate_counts(
                    outputs_sig.detach(), targets.detach(), mask.detach())
                tp_sum += tp
                fp_sum += fp
                fn_sum += fn
                tn_sum += tn
                y_true_all.append(y_true)
                y_prob_all.append(y_prob)

            loss_sum += loss_aggregated.sum().item()
            mask_count_sum += mask.sum().item()
            batch_itr += 1

            optimizer.zero_grad()
            (loss_avg + l2_loss * 0.01).backward()
            # (loss_avg).backward()
            optimizer.step()

            # --- TensorBoard per-step logs ---
            writer.add_scalar("train/loss_batch", loss_avg.item(), global_step)
            writer.add_scalar("train/l2_loss_batch", float(l2_loss.item()), global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], global_step)

            if batch_idx % 10 == 0:  # Log every 10 batches
                print(f'Train Epoch {epoch}, Batch {batch_idx}, Loss: {loss_avg.item()}')
                logging.info(f'Train Epoch {epoch}, Batch {batch_idx}, Loss: {loss_avg.item()}, Accuracy: {m["accuracy"]}')

            global_step += 1


        # --- Epoch-level aggregates ---
        train_loss = loss_sum / max(mask_count_sum, 1)
        avg_metrics = compute_metrics_from_counts(
            tp_sum, fp_sum, fn_sum, tn_sum,
            y_true=np.concatenate(y_true_all) if y_true_all else None,
            y_prob=np.concatenate(y_prob_all) if y_prob_all else None,
        )

        print(f'Epoch {epoch}, Training Loss: {train_loss}')
        logging.info(f'Epoch {epoch}, Training Loss: {train_loss}, '
                     f'Acc: {avg_metrics["accuracy"]}, P: {avg_metrics["precision"]}, '
                     f'R: {avg_metrics["recall"]}, R: {avg_metrics["acc_free"]}, F1: {avg_metrics["f1"]}')

        writer.add_scalar("train/loss_epoch", train_loss, epoch)
        writer.add_scalar("train/accuracy_epoch", avg_metrics["accuracy"], epoch)
        writer.add_scalar("train/precision_epoch", avg_metrics["precision"], epoch)
        writer.add_scalar("train/recall_epoch", avg_metrics["recall"], epoch)
        writer.add_scalar("train/acc_free_epoch", avg_metrics["acc_free"], epoch)
        writer.add_scalar("train/f1_epoch", avg_metrics["f1"], epoch)
        if avg_metrics["roc_auc"] is not None:
            writer.add_scalar("train/roc_auc_epoch", avg_metrics["roc_auc"], epoch)

        # Validate the model
        valid_loss = evaluate(model, valid_data_loader, config['model']["device"], writer, epoch)
        print(f'Epoch {epoch}, Validation Loss: {valid_loss}')
        logging.info(f'Epoch {epoch}, Validation Loss: {valid_loss}')

        # Save model checkpoint
        if save_ckpt:
            model_path = config['model']['model_output_path']
            torch.save(model.state_dict(), model_path.format(epoch))

        if valid_loss < best_loss:
            best_loss = valid_loss
            if save_best_ckpt:
                best_path = os.path.join(os.path.dirname(config['model']['model_output_path']), "best.pt")
                torch.save(model.state_dict(), best_path)
            print(f'New best model saved at epoch {epoch} with validation loss {best_loss}')
            logging.info(f'New best model saved at epoch {epoch} with validation loss {best_loss}')

        if config['model']['use_lr_scheduler']:
            lr_scheduler.step()
            print(f'Learning rate adjusted to: {lr_scheduler.get_last_lr()[0]}')

    writer.close()

    # Save the final model checkpoint
    if save_ckpt:
        torch.save(model.state_dict(), config['model']['model_output_path'].format('final'))

    return best_loss


if __name__ == '__main__':
    with open("../configs/config.yaml", "r") as stream:
        config = yaml.safe_load(stream)
        config['data']['batch_size'] = config['model']['train_batch_size']

    train_dataloader = OGMDataLoader(config['data'], phase='train').create_dataloader()
    valid_dataloader = OGMDataLoader(config['data'], phase='validation').create_dataloader()

    model = VSTSBGT(config['model']).to(config['model']["device"])
    # model.load_state_dict(torch.load(config['model']['model_output_path'].format(98)))
    # model = model.to(config['model']["device"])

    train(model, train_dataloader, valid_dataloader, config)
