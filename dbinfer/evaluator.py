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


import torch
import torch.nn.functional as F
import torchmetrics.functional as MF
import torchmetrics.retrieval as MR
from dbinfer_bench import DBBTaskType, DBBTaskMeta, DBBGraphDataset
from dgl import to_networkx
from networkx import Graph
from typing import List, Tuple

negated = lambda f: lambda *args, **kwargs: -f(*args, **kwargs)


def root_mean_squared_error(logits, target):
    return MF.mean_squared_error(logits, target, squared=False)


METRIC_FN = {
    "classification": {
        "accuracy": MF.accuracy,
        "ap": MF.average_precision,
        "auroc": MF.auroc,
        "f1": MF.f1_score,
        "hinge": negated(MF.hinge_loss),
        "recall": MF.recall,
    },
    "regression": {
        "mae": negated(MF.mean_absolute_error),
        "mse": negated(MF.mean_squared_error),
        "msle": negated(MF.mean_squared_log_error),
        "pearson": MF.pearson_corrcoef,
        "rmse": negated(root_mean_squared_error),
        "r2": MF.r2_score,
    },
    "retrieval": {
        "hr": MR.RetrievalHitRate(),
        "mrr": MR.RetrievalMRR(),
        "ndcg": MR.RetrievalNormalizedDCG(),
    },
}

def get_metric_fn(meta : DBBTaskMeta):
    fn = METRIC_FN[meta.task_type][meta.evaluation_metric]
    if meta.task_type == DBBTaskType.classification:
        def _classification_wrapper(seeds, logits, labels):
            # Shape:
            #   - logits: (N, C) or (N,) if C == 1
            #   - labels : (N,)
            with torch.no_grad():
                preds = F.softmax(logits, dim=1)
                return fn(preds, labels,
                          num_classes=meta.num_classes,
                          task='multiclass')
        return _classification_wrapper
    elif meta.task_type == DBBTaskType.regression:
        def _regression_wrapper(seeds, logits, targets):
            # Shape:
            #   - logits: (N,)
            #   - targets : (N,)
            with torch.no_grad():
                return fn(logits, targets.float())
        return _regression_wrapper
    elif meta.task_type == DBBTaskType.retrieval:
        def _retrieval_wrapper(query_idx, logits, labels):
            # Shape:
            #   - query_idx: (N,)
            #   - logits: (N,)
            #   - labels : (N,)
            with torch.no_grad():
                preds = torch.sigmoid(logits)
                return fn(preds, labels, indexes=query_idx)
        return _retrieval_wrapper
    else:
        raise ValueError(f"Unsupported task type {meta.task_type}")

def infer_task_type(metric_name: str):
    for task_type, metrics in METRIC_FN.items():
        if metric_name in metrics:
            return task_type
    raise ValueError(f"Invalid metric name {metric_name}.")

def retrieval_loss(logits, labels):
    return F.binary_cross_entropy(torch.sigmoid(logits), labels.float())

LOSS_FN = {
    'classification': F.cross_entropy,
    'regression': lambda logits, targets : F.mse_loss(logits, targets.float()),
    'retrieval': retrieval_loss,
}

def get_loss_fn(meta : DBBTaskMeta):
    fn = LOSS_FN[meta.task_type]
    return fn


def louvain(edges: List[Tuple]):
    import community as community_louvain
    partition = community_louvain.best_partition(Graph(edges))


def leiden(edges: List[Tuple]):
    import igraph as ig
    import leidenalg
    g = ig.Graph.TupleList(edges)
    partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
    return partition

import numpy as np
from numba import njit

@njit
def compute_homophily_numba(edges, labels):
    """
    edges: 2D numpy array (E, 2), int64
    labels: 1D numpy array (N,), int64
    return: float, homophily ratio
    """
    same = 0
    E = edges.shape[0]
    for i in range(E):
        u = edges[i, 0]
        v = edges[i, 1]
        if labels[u] == labels[v]:
            same += 1
    return same / E if E > 0 else 0.0


# # Example
# edges = np.array([(0,1), (1,2), (2,3), (3,4), (4,0)], dtype=np.int64)
# labels = np.array([0,0,1,1,0], dtype=np.int64)

# print(compute_homophily_numba(edges, labels))  

def compute_homophily(edges, labels):
    """
    edges: List[Tuple[int, int]]
    labels: List[int]
    return: float, homophily ratio
    """
    edges = np.asarray(edges, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)

    # 获取边两端节点的标签
    u_labels = labels[edges[:, 0]]
    v_labels = labels[edges[:, 1]]

    # 判断是否同类
    same = (u_labels == v_labels)

    return same.sum() / len(edges) if len(edges) > 0 else 0.0