import argparse
import copy
import os
import shutil

import optuna
import yaml

from dataset.data_loader import OGMDataLoader
from models.v_stsbgat import VSTSBGT
from train import train

ENCODER_DIM_CHOICES = [16, 32, 64, 128]
CELL_DIM_CHOICES = [16, 32, 64, 128]
GAT_DIM_CHOICES = [16, 32, 64, 128]
HEAD_CHOICES = [1, 2, 4, 8]  # divides every value in the *_DIM_CHOICES lists evenly


def suggest_params(trial):
    return {
        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'encoder_dim_model': trial.suggest_categorical('encoder_dim_model', ENCODER_DIM_CHOICES),
        'obs_num_heads': trial.suggest_categorical('obs_num_heads', HEAD_CHOICES),
        'z_num_heads': trial.suggest_categorical('z_num_heads', HEAD_CHOICES),
        'obs_num_layers': trial.suggest_int('obs_num_layers', 1, 6),
        'z_num_layers': trial.suggest_int('z_num_layers', 1, 6),
        'obs_dropout': trial.suggest_float('obs_dropout', 0.0, 0.5),
        'z_dropout': trial.suggest_float('z_dropout', 0.0, 0.5),
        'cell_dim': trial.suggest_categorical('cell_dim', CELL_DIM_CHOICES),
        'gat_dim_model': trial.suggest_categorical('gat_dim_model', GAT_DIM_CHOICES),
        'gat_num_heads': trial.suggest_int('gat_num_heads', 1, 8),
        'gat_dropout': trial.suggest_float('gat_dropout', 0.0, 0.5),
        'map_projection_dim': trial.suggest_categorical('map_projection_dim', [16, 32, 64]),
        'map_context_dim': trial.suggest_categorical('map_context_dim', [16, 32, 64, 128]),
        'map_dropout': trial.suggest_float('map_dropout', 0.0, 0.5),
    }


def apply_params(base_config, params):
    """Applies a flat hyperparameter dict (from suggest_params(), or from
    study.best_params) onto a fresh copy of the base config. Kept separate
    from suggest_params() because study.best_params is a plain dict, not a
    live Trial, so it can't call trial.suggest_*() again."""
    config = copy.deepcopy(base_config)
    m = config['model']
    m['lr'] = params['lr']
    m['obs_temporal_encoder']['dim_model'] = params['encoder_dim_model']
    m['z_temporal_encoder']['dim_model'] = params['encoder_dim_model']
    m['gat']['dim_obs_model'] = params['encoder_dim_model']
    m['obs_temporal_encoder']['num_heads'] = params['obs_num_heads']
    m['z_temporal_encoder']['num_heads'] = params['z_num_heads']
    m['obs_temporal_encoder']['num_layers'] = params['obs_num_layers']
    m['z_temporal_encoder']['num_layers'] = params['z_num_layers']
    m['obs_temporal_encoder']['dropout'] = params['obs_dropout']
    m['z_temporal_encoder']['dropout'] = params['z_dropout']
    m['map_encoder']['cell_node_dim'] = params['cell_dim']
    m['gat']['dim_cell_model'] = params['cell_dim']
    m['gat']['dim_model'] = params['gat_dim_model']
    m['gat']['num_heads'] = params['gat_num_heads']
    m['gat']['dropout'] = params['gat_dropout']
    m['map_encoder']['projection_dim'] = params['map_projection_dim']
    m['map_encoder']['map_context_dim'] = params['map_context_dim']
    m['map_encoder']['dropout'] = params['map_dropout']
    return config


def build_trial_config(base_config, trial, results_dir, epochs_override):
    config = apply_params(base_config, suggest_params(trial))
    if epochs_override is not None:
        config['model']['train_epochs'] = epochs_override

    # Trial-isolated output paths so trials never collide on checkpoints/logs.
    trial_dir = os.path.join(results_dir, f"trial_{trial.number}")
    os.makedirs(trial_dir, exist_ok=True)
    config['model']['model_output_path'] = os.path.join(trial_dir, "epoch_{}.pt")
    config['model']['tb_log_dir'] = os.path.join(trial_dir, "runs")
    config['model']['log_file'] = os.path.join(trial_dir, "app_{}.log")
    config['model']['save_checkpoints'] = False  # skip per-epoch saves (only the metric matters)
    config['model']['save_best_checkpoint'] = True  # but keep this trial's single best-epoch save

    config['data']['batch_size'] = config['model']['train_batch_size']
    return config


def keep_only_overall_best_checkpoint(trial, trial_value, results_dir):
    """Keeps at most ONE checkpoint on disk across the whole study - the current
    best-overall trial's weights - rather than one per trial (which balloons disk usage
    over many trials). Compares this trial's value against the best of all PREVIOUSLY
    completed trials (this trial isn't finalized in the study yet while its own
    objective() call is still running), promotes this trial's checkpoint to the stable
    path if it's a new best, and always deletes the trial-local copy afterward.
    """
    trial_dir = os.path.join(results_dir, f"trial_{trial.number}")
    trial_ckpt = os.path.join(trial_dir, "best.pt")
    if not os.path.exists(trial_ckpt):
        return

    stable_best_path = os.path.normpath(
        os.path.join(results_dir, "..", "checkpoints", "best_optuna_trial.pt"))

    prior_values = [t.value for t in trial.study.get_trials(deepcopy=False)
                   if t.state == optuna.trial.TrialState.COMPLETE]
    prior_best = min(prior_values) if prior_values else None

    if prior_best is None or trial_value <= prior_best:
        os.makedirs(os.path.dirname(stable_best_path), exist_ok=True)
        shutil.copy2(trial_ckpt, stable_best_path)
        print(f"Trial {trial.number} (value={trial_value:.6f}) is the new best - "
             f"checkpoint saved to {stable_best_path}")
    else:
        print(f"Trial {trial.number} (value={trial_value:.6f}) did not beat the current "
             f"best ({prior_best:.6f}) - discarding its checkpoint to save disk space")

    os.remove(trial_ckpt)  # never keep more than the single promoted checkpoint on disk


def objective(trial, base_config, results_dir, epochs_override):
    config = build_trial_config(base_config, trial, results_dir, epochs_override)

    train_dataloader = OGMDataLoader(config['data'], phase='train').create_dataloader()
    valid_dataloader = OGMDataLoader(config['data'], phase='validation').create_dataloader()
    model = VSTSBGT(config['model']).to(config['model']["device"])

    best_loss = train(model, train_dataloader, valid_dataloader, config)
    keep_only_overall_best_checkpoint(trial, best_loss, results_dir)
    return best_loss


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for VSTSBGT")
    parser.add_argument('--n_trials', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=None,
                        help="Override train_epochs for every trial (e.g. for a fast smoke test).")
    parser.add_argument('--study_name', type=str, default='vstsbgat_hpo')
    parser.add_argument('--storage', type=str, default='sqlite:///../results/optuna_study.db')
    args = parser.parse_args()

    with open("../configs/config.yaml", "r") as stream:
        base_config = yaml.safe_load(stream)

    results_dir = "../results/optuna"
    os.makedirs(results_dir, exist_ok=True)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(trial, base_config, results_dir, args.epochs),
        n_trials=args.n_trials,
        catch=(RuntimeError,),  # e.g. CUDA OOM on a large dim_model draw shouldn't kill the sweep
    )

    print(f"Best value (validation loss): {study.best_value}")
    print(f"Best params: {study.best_params}")

    best_config = apply_params(base_config, study.best_params)
    out_path = "../configs/config_all_scenes_convnext.yaml"
    with open(out_path, "w") as f:
        yaml.safe_dump(best_config, f)
    print(f"Best config written to {out_path}")

    # keep_only_overall_best_checkpoint() already keeps this up to date after every trial -
    # nothing left to copy here, just report whether one exists.
    stable_best_path = os.path.normpath(
        os.path.join(results_dir, "..", "checkpoints", "best_optuna_trial.pt"))
    if os.path.exists(stable_best_path):
        print(f"Best-known checkpoint available at {stable_best_path}")
    else:
        print(f"No checkpoint available at {stable_best_path} - the current best trial "
             f"(trial {study.best_trial.number}, value={study.best_value}) predates "
             f"checkpoint saving and has no saved weights.")


if __name__ == '__main__':
    main()
