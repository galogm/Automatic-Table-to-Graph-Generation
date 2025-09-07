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

from utils.data.rdb import load_dbb_dataset, load_dbb_dataset_from_cfg_path

from ..device import get_device_info
from ..preprocess import (
    get_rdb_preprocess_class,
    get_rdb_preprocess_choice,
)
from .. import yaml_utils

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

RDBPreprocessChoice = get_rdb_preprocess_choice()

def preprocess(
    dataset_name: str = typer.Argument(
        ...,
        help="The dataset name of the RDB dataset"
    ),
    dataset_path : str = typer.Argument(
        ...,
        help=("Path to the dataset folder or one of the built-in datasets. "
              "Use the list-builtin command to list all the built-in datasets.")
    ),
    preprocess_name : str = typer.Argument(
        ...,
        help="Preprocess name"
    ),
    output_path : str = typer.Argument(
        ..., 
        help="Output path for the preprocessed dataset."
    ),
    config_path : str = typer.Argument(
        None,
        help="Solution configuration path. Use default if not specified."
    )
):
    output_path = Path(output_path)
    device = get_device_info()
    logger.debug(device)
    preprocess_class = get_rdb_preprocess_class(preprocess_name)
    if config_path is None:
        logger.info("No solution configuration file provided. Use default configuration.")
        config = preprocess_class.default_config
    else:
        logger.info(f"Load solution configuration file: {config_path}.")
        config = yaml_utils.load_pyd(preprocess_class.config_class, config_path)

    logger.debug(f"Config:\n{config.json()}")

    logger.info("Loading data ...")
    # ataset = dbb.load_rdb_data(dataset_path)
    # dataset, _ = load_dbb_dataset(dataset_name, dataset_path)
    dataset, _ = load_dbb_dataset_from_cfg_path(dataset_name, dataset_path)

    logger.info("Creating preprocess ...")
    preprocess = preprocess_class(config)

    preprocess.run(dataset, output_path, device)
    
    return output_path
