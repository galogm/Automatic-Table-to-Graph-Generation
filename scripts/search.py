"""Parameter Search:
```bash
log_dir=logs/movielens/autom/;id=0;d=MVLS;gpu=$id;model=sage;task=ratings;method=r2n;use_cache=1;mkdir -p $log_dir;nohup python -u -m scripts.search --gpu=$gpu --model $model --task $task --method $method --use_cache $use_cache --dataset=$d --n_trials=20 --n_jobs=2 > $log_dir/$method-$model-$task-$id.log 2>&1 & echo $!
```
"""

import argparse
import gc
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Literal, Union, Dict

import optuna
import torch
from optuna.trial import TrialState
import yaml
import traceback
from utils.rdb import name_id_mapping

logger = optuna.logging.get_logger("optuna")


class PaddedLevelFormatter(logging.Formatter):
    """Formatter that supports microseconds in datefmt."""

    MAX_LEVEL_LENGTH = 8

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created)

        if datefmt:
            return dt.strftime(datefmt)

        return f"{dt.strftime('%Y-%m-%d %H:%M:%S')},{record.msecs:03d}"

    def format(self, record):
        record.levelname = record.levelname.ljust(self.MAX_LEVEL_LENGTH)
        return super().format(record)


# fmt: off
search_space = {
    "lr": [0.001, 0.01, 0.005],
    "batch_size": [2048, 4196, 8196],
    "hid_size": [64, 128, 256, 512],
    "feat_encode_size": [64, 128, 256, 512],
    "fanouts": [[25, 20], [30, 25], [35, 30]],
    "negative_sampling_ratio": [5, 10, 15],
    "patience": [5, 10, 15, 20],
    "epochs": [100, 150, 200],
    "time_budget": 36000,
    "eval_trials": [1, 3, 5, 10],
    "dropout": [0, 0.2, 0.5, 0.8, 0.9],
    "pred_num_layers": [1, 2, 3, 4],
    "pred_hid_size": [64, 128, 256, 512],
    "pred_dropout": [0, 0.2, 0.5, 0.8, 0.9],
}
# fmt: on


class GPUOutOfMemoryError(torch.cuda.OutOfMemoryError):
    """Custom exception for GPU Out-of-Memory errors."""


class NaNError(ValueError):
    """Custom exception for GPU Out-of-Memory errors."""


class ValueNoneError(ValueError):
    """Return None."""


def wait_for_study_completion(study, poll_interval=60):
    """Wait until all Optuna trials are either COMPLETE or FAIL."""
    cnt = 0
    while True:
        trials = study.trials
        total = len(trials)
        complete = sum(t.state == optuna.trial.TrialState.COMPLETE for t in trials)
        pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in trials)
        fail = sum(t.state == optuna.trial.TrialState.FAIL for t in trials)
        running = sum(t.state == optuna.trial.TrialState.FAIL for t in trials)
        if total == complete + fail + pruned or running == 0:
            return total, complete, fail
        time.sleep(poll_interval)
        cnt = cnt + 1
        time_waited = cnt * poll_interval / 60
        logger.info(f"wait {time_waited} mins")
        if time_waited > 360:
            return None, None, None


def should_retry(flag_file_path: Union[str, Path]) -> bool:
    """
    Checks a one-time flag.
    """
    if os.path.exists(flag_file_path):
        return False
    fd = -1
    try:
        fd = os.open(flag_file_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        logger.info(f"Retry in process {os.getpid()}")
        return True
    except OSError:
        logger.info(f"Not Retry in process {os.getpid()}")
        return False
    finally:
        if fd != -1:
            try:
                os.close(fd)
            except OSError:
                pass


def save_yaml(yaml_path: Path, params_config):
    try:
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(params_config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Config Saved: {yaml_path}")
    except Exception as e:
        logger.error(f"Generating YAML Error: {e}")
        traceback.print_exc()
        raise RuntimeError(f"Generating YAML Error: {e}")


def objective(
    trial: optuna.trial.Trial,
    dataset: str,
    tsk: str,
    method: str,
    gnn: str,
    use_cache: int,
    log_path: Path,
    path_cache: Path,
    path_schema: Path,
    prune_fail_pruned=True,
):
    logger = optuna.logging.get_logger("optuna")

    trial_id = trial.number

    # fmt: off
    params_config = {}
    predictor_config = {}

    for key, values in search_space.items():
        if not isinstance(values, list):
            chosen_value = values
        else:
            chosen_value = trial.suggest_categorical(key, values)

        if key.startswith("pred_"):
            clean_key = key.replace("pred_", "", 1)
            predictor_config[clean_key] = chosen_value
        else:
            params_config[key] = chosen_value

    params_config["eval_fanouts"] = params_config["fanouts"]
    params_config["eval_batch_size"] = params_config["batch_size"]

    if predictor_config:
        params_config["predictor"] = predictor_config

    yaml_filename = f"{trial_id}.yaml"

    path_config = Path(f"{log_path}/{yaml_filename}")
    path_config.parent.mkdir(exist_ok=True, parents=True)
    save_yaml(path_config, params_config)

    cmd = [
        "python3",
        "-u",
        "-m",
        "main.rdb",
        f"{dataset}",
        f"{path_cache}",
        f"{path_schema}",
        f"{method}",
        f"{gnn}",
        f"{tsk}",
        "-c",
        f"{path_config}",
        "--use-cache",
        f"{use_cache}",
    ]
    # fmt: on

    logger.info(f"Trial {trial_id}: {' '.join(cmd)}")

    states_to_consider = [TrialState.COMPLETE]
    if prune_fail_pruned:
        states_to_consider.extend([TrialState.FAIL, TrialState.PRUNED])
    trials_to_consider = trial.study.get_trials(
        deepcopy=False, states=tuple(states_to_consider)
    )

    # Prune duplicated trail settings
    for t in reversed(trials_to_consider):
        if trial.params == t.params:
            logger.info(f"Duplicate trial pruned: {trial.params}")
            raise optuna.exceptions.TrialPruned()

    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
        )

        if (
            "CUDA out of memory" in result.stderr
            or "CUDA out of memory" in result.stdout
        ):
            logger.error("Trial %d encountered OOM. FAIL.", trial.number)
            trial.set_user_attr("fail_reason", "GPU_OOM")
            original_oom_trial_number = trial.user_attrs.get(
                "original_oom_trial_number", trial.number
            )
            trial.set_user_attr("original_oom_trial_number", original_oom_trial_number)
            raise GPUOutOfMemoryError("CUDA OOM error detected in subprocess.")

        if "contains NaN" in result.stderr or "contains NaN" in result.stdout:
            logger.error("Trial %d encountered NaN. FAIL.", trial.number)
            trial.set_user_attr("fail_reason", "NaN_value")
            original_oom_trial_number = trial.user_attrs.get(
                "original_nan_trial_number", trial.number
            )
            trial.set_user_attr("original_nan_trial_number", original_oom_trial_number)
            raise NaNError("NaN error detected in subprocess.")

        if result.returncode != 0:
            logger.error(
                "Trial %d: Subprocess exited with error code %d.",
                trial.number,
                result.returncode,
            )
            trial.set_user_attr(
                "fail_reason", f"SUBPROCESS_EXIT_CODE_{result.returncode}"
            )
            raise RuntimeError(
                f"Subprocess failed with exit code {result.returncode}. Stderr: {result.stderr.strip()}"
            )

        metric_to_optimize = "Test metric:"

        metric_match = re.search(
            rf"\s*{re.escape(metric_to_optimize)}\s*(\d+\.\d+)\s*", result.stdout
        ) or re.search(
            rf"\s*{re.escape(metric_to_optimize)}\s*(\d+\.\d+)\s*", result.stderr
        )

        metric_value = None
        if metric_match:
            mean_str = metric_match.group(1)
            metric_value = float(mean_str)
        if metric_value is not None:
            logger.info(f"Trial {trial_id} for dataset {dataset}: {metric_value}")
            return -metric_value

        logger.error("Trial %d encountered None Value. FAIL.", trial.number)
        raise ValueNoneError(f"Trial {trial_id} failed with value None.")

    finally:
        torch.cuda.empty_cache()
        gc.collect()
        with open(log_path / f"{trial_id}.log", "w", encoding="utf-8") as f:
            f.write("CMD:\n" + " ".join(cmd) + "\n\n")
            f.write("STDOUT:\n" + result.stdout + "\n\n")
            f.write("STDERR:\n" + result.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run your script with specified GPU, metric, dataset, trials, and jobs.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--gpu", type=str, default="0", help="The GPU number to use (e.g., 0, 1)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="The model type to use",
    )
    parser.add_argument(
        "--task",
        type=str,
        help="The task type to use",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["r2n", "r2ne"],
        help="The method type to use",
    )
    parser.add_argument(
        "--use_cache",
        type=int,
        default=1,
        help="Whether to use caching",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(name_id_mapping.keys()),
        help="The name of the dataset to process",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=256,
        help="Number of trials for the experiment (default: 256)",
    )
    parser.add_argument(
        "--n_jobs", type=int, default=1, help="Number of parallel jobs (default: 1)"
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Access the parsed arguments
    d = name_id_mapping[args.dataset]
    model = args.model
    method = args.method
    task = args.task
    use_cache = args.use_cache
    n_trials: int = args.n_trials
    n_jobs: int = args.n_jobs
    MODEL: str = f"{method}_{model}_{d}_{task}"

    LOCK_FILE_FLOCK = Path(f"logs/{d}/autom/{method}/{model}/{task}/lock")
    log_path = LOCK_FILE_FLOCK.parent
    log_path.mkdir(exist_ok=True, parents=True)

    path_cache = Path(f"./data/{d}/autom/{task}/final")
    path_schema = path_cache
    path_cache.parent.mkdir(exist_ok=True, parents=True)

    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{log_path}/hyperparams.db",
        engine_kwargs={"connect_args": {"timeout": 100}},
    )
    study = optuna.create_study(
        direction="minimize",
        study_name=f"{MODEL}_train_study",
        storage=storage,
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(
            trial,
            dataset=args.dataset,
            tsk=task,
            method=method,
            gnn=model,
            use_cache=use_cache,
            log_path=log_path,
            path_cache=path_cache,
            path_schema=path_schema,
        ),
        n_trials=n_trials,
        n_jobs=n_jobs,
        catch=(
            GPUOutOfMemoryError,
            NaNError,
            optuna.exceptions.TrialPruned,
            ValueNoneError,
        ),
        gc_after_trial=True,
    )

    if should_retry(LOCK_FILE_FLOCK):
        total, complete, fail = wait_for_study_completion(study=study)

        if fail is not None:
            logger.info(
                f"Trails: {total}",
            )
            logger.info(f"Complete: {complete}")
            logger.info(f"Failed: {fail}")

            trials_to_check = study.get_trials(deepcopy=True)
            trials_enqueued = 0
            for trial in trials_to_check:
                if (
                    trial.state != optuna.trial.TrialState.FAIL
                    or trial.user_attrs.get("fail_reason") != "GPU_OOM"
                ):
                    continue
                study.enqueue_trial(
                    params=trial.params,
                    user_attrs={
                        "is_retried_oom": True,
                        "original_oom_trial_number": trial.number,
                        "is_retry": True,
                    },
                )
                trials_enqueued += 1

            if trials_enqueued > 0:
                logger.info(f"Enqueued {trials_enqueued} trials for retry.")
                study.optimize(
                    lambda trial: objective(
                        trial,
                        dataset=args.dataset,
                        tsk=task,
                        method=method,
                        gnn=model,
                        use_cache=use_cache,
                        log_path=log_path,
                        path_cache=path_cache,
                        path_schema=path_schema,
                        prune_fail_pruned=False,
                    ),
                    n_trials=trials_enqueued,
                    n_jobs=1,
                    catch=(
                        GPUOutOfMemoryError,
                        NaNError,
                        optuna.exceptions.TrialPruned,
                        ValueNoneError,
                    ),
                    gc_after_trial=True,
                )

            logger.info(
                f"""
--- Final Study Results ---
Total number of trials: {len(study.trials)}
Total complete trials: {len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]))}
Total failed trials: {len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.FAIL]))}
"""
            )

    logger.info(f"Best trial: {study.best_trial._trial_id}")
    logger.info(f"Value: {study.best_value}")
    logger.info(f"Params: {study.best_params}")
