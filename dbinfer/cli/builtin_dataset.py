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


import typer
import dbinfer_bench as dbb
from utils import logger

def list_builtin():
    logger.info('\n'.join(dbb.list_builtin()))

def download(
    dataset_name : str = typer.Argument(
        ...,
        help="Dataset name to download."
    ),
    version : str = typer.Option(
        None,
        help="Dataset version."
    ),
):
    dataset_path = dbb.get_builtin_path_or_download(dataset_name, version)
    logger.info(f"Dataset downloaded to '{dataset_path}'.")
