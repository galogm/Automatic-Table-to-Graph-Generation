# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.


from pathlib import Path
import typer
import logging
import wandb
import os
import numpy as np

import dbinfer_bench as dbb

from ..device import DeviceInfo
from ..solutions import (
    get_gml_solution_class,
    parse_config_from_graph_dataset,
    get_gml_solution_choice,
)
from .. import yaml_utils
from .fit_utils import _fit_main

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

GMLSolutionChoice = get_gml_solution_choice()


def replace_hyperparameters_for_search(config, **kwargs):
    pass

def fit_gml(
    dataset_path : str = typer.Argument(
        ...,
        help=("Path to the dataset folder or one of the built-in datasets. "
              "Use the list-builtin command to list all the built-in datasets.")
    ),
    task_name : str = typer.Argument(
        ...,
        help=("Name of the task to fit the solution.")
    ),
    solution_name : GMLSolutionChoice = typer.Argument(
        ...,
        help="Solution name"
    ),
    config_path : Path = typer.Option(
        None,
        "--config_path", "-c",
        help="Solution configuration path. Use default if not specified."
    ),
    checkpoint_path : str = typer.Option(
        None, 
        "--checkpoint_path", "-p",
        help="Checkpoint path."
    ),
    enable_wandb : bool = typer.Option(
        True,
        "--enable-wandb/--disable-wandb",
        help="Enable Weight&Bias for logging."
    ),
    num_runs : int = typer.Option(
        1,
        "--num-runs", "-n",
        help="Number of runs."
    ), 
    sweep: bool = typer.Option(
        False,
        "--hypertune",
        help="Whether to use hyperparameter tuning."
    )
):
    solution_class = get_gml_solution_class(solution_name.value)
    if config_path == None:
        ## sweep_mode
        if sweep:
            logger.info("Use wandb to do the hyperparameter search.")   
            solution_config = None
        else:
            logger.info("No solution configuration file provided. Use default configuration.")
            solution_config = solution_class.config_class()
    else:
        logger.info(f"Load solution configuration file: {config_path}.")
        solution_config = yaml_utils.load_pyd(solution_class.config_class, config_path)

    
    # logger.debug(f"Solution config:\n{solution_config.json()}")

    logger.info("Loading data ...")
    dataset = dbb.load_graph_data(dataset_path)

        
    data_config = parse_config_from_graph_dataset(dataset, task_name)
    logger.debug(f"Data config:\n{data_config.json()}")
    def _invoke_fit(solution, run_ckpt_path : Path, device : DeviceInfo):
        summary = solution.fit(dataset, task_name, run_ckpt_path, device)
        return summary

    def _invoke_test(solution, run_ckpt_path : Path, device : DeviceInfo):
        solution.load_from_checkpoint(run_ckpt_path)
        val_metric = solution.evaluate(
            dataset.graph_tasks[task_name].validation_set,
            dataset.graph,
            dataset.feature,
            device,
        )
        test_metric = solution.evaluate(
            dataset.graph_tasks[task_name].test_set,
            dataset.graph,
            dataset.feature,
            device,
            is_test=True
        )
        return val_metric, test_metric

    train_metric, val_metric, test_metric = _fit_main(
        solution_class,
        dataset,
        data_config,
        solution_config,
        checkpoint_path,
        enable_wandb,
        num_runs,
        _invoke_fit,
        _invoke_test
    )
    
    wandb.log({"val_metric": val_metric})
    
    return val_metric, test_metric
