"""Evaluation."""

from typing import List, Tuple

import community as community_louvain
import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
from networkx import Graph
from numba import njit
from scipy.stats import mode
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    roc_auc_score,
)

from .logging import get_logger

logger = get_logger(__name__)


def _compute_binary_community_homophily(true_labels, comm_labels, pos_label=1):
    """
    Compute homophily for each community in a binary setting.
    Homophily = fraction of nodes in the community belonging to the majority label (0 or 1).
    Returns np.ndarray of homophily per community.
    """
    homophilies = []
    nums = []
    for c in np.unique(comm_labels):
        idx = np.where(comm_labels == c)[0]
        if len(idx) == 0:
            homophilies.append(-1)
            continue
        labels_in_comm = true_labels[idx]
        nums.append(len(labels_in_comm))
        pos_ratio = np.sum(labels_in_comm == pos_label) / nums[-1]
        homophilies.append(pos_ratio)
    return np.array(homophilies), [n / len(comm_labels) for n in nums]


def eval_leiden_louvain(
    label_path: str = "./data/retailrocket/autog/cvr/final/data/View.pqt",
    edge_path: str = "./data/retailrocket/autom/cvr/final/data/View_Visitor_View_ItemCategory_Category_ItemCategory_View.pqt",
    method: str = "leiden",
    element_col: str = "itemid",
    label_col: str = "added_to_cart",
    head_col: str = "head_itemid",
    tail_col: str = "tail_itemid",
    directed: bool = False,
):
    """Run Leiden and Louvain and compute NMI, ARI, ACC, and community homophily."""
    ids, labels = get_labels(label_path, element_col, label_col)
    heads, tails = get_edges(edge_path, head_col, tail_col)
    # edges, aligned_labels = _align_nodes(heads, tails, ids, labels)
    edges, aligned_labels, nodes = _align_node_indices(heads, tails, ids, labels)

    if method.lower() == "leiden":
        g = ig.Graph.TupleList(edges.tolist(), directed=directed)
        partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
        comm_labels = np.array(partition.membership, dtype=int)
    elif method.lower() == "louvain":
        G = Graph(edges.tolist(), directed=directed)
        partition = community_louvain.best_partition(G)
        logger.info(
            "Partition: %s\nNodes: %s\nLabels: %s",
            f"{partition}",
            f"{len(nodes)}/{len(ids)}",
            f"{labels}",
        )
        comm_labels = np.array([partition.get(i, -1) for i in range(len(nodes))], dtype=int)
    else:
        raise ValueError("method must be 'leiden' or 'louvain'")

    y_true = aligned_labels[: len(comm_labels)]
    nmi = normalized_mutual_info_score(y_true, comm_labels)
    ari = adjusted_rand_score(y_true, comm_labels)
    acc = _compute_acc(y_true, comm_labels)
    # comm_homo = _compute_community_homophily(y_true, comm_labels)
    comm_homo_pos, nums = _compute_binary_community_homophily(y_true, comm_labels, 1)
    comm_homo_neg, _ = _compute_binary_community_homophily(y_true, comm_labels, 0)
    comm_homo_pos[comm_homo_pos == -1] = 0
    comm_homo_neg[comm_homo_neg == -1] = 0

    # For AUC-ROC, use the fraction of positive nodes per community as predicted score
    scores = np.zeros_like(y_true, dtype=float)
    for c in np.unique(comm_labels):
        idx = np.where(comm_labels == c)[0]
        labels_in_comm = y_true[idx]
        score = np.mean(labels_in_comm == 1)
        scores[idx] = score
    try:
        aucroc = roc_auc_score(y_true, scores)
    except ValueError:
        aucroc = np.nan  # e.g., if only one class exists

    return {
        "NMI": nmi,
        "ACC": acc,
        "ARI": ari,
        "AUCROC": aucroc,
        "POS_COMM_HOMO_ARRAY": comm_homo_pos,
        "NEG_COMM_HOMO_ARRAY": comm_homo_neg,
        "POS_COMM_HOMO_MEAN": comm_homo_pos.mean(),
        "NEG_COMM_HOMO_MEAN": comm_homo_neg.mean(),
        "COMM_RATIO": nums,
    }


def _align_node_indices(heads, tails, ids, labels):
    """Map arbitrary node IDs to contiguous indices and align edges and labels."""
    unique_nodes = np.unique(np.concatenate([heads, tails]))
    id_to_idx = {nid: i for i, nid in enumerate(unique_nodes)}
    edges = np.vectorize(id_to_idx.get)(np.stack([heads, tails], axis=1))

    # Align labels for nodes present in edges
    label_map = {nid: lab for nid, lab in zip(ids, labels) if nid in id_to_idx}
    aligned_labels = np.array([label_map.get(nid, -1) for nid in unique_nodes], dtype=np.int64)
    valid_mask = aligned_labels != -1
    edges = edges[np.all(valid_mask[edges], axis=1)]
    aligned_labels = aligned_labels[valid_mask]

    logger.info(
        "Node Num: %s, Label Num: %s, Edge Num: %s, Com: %s/%s %s %s/%s %s",
        f"{len(unique_nodes)}",
        f"{len(np.unique(aligned_labels))}",
        f"{len(edges)}",
        f"{min(ids)}",
        f"{max(ids)}",
        f"{sorted(np.unique(ids))[0:500]}",
        f"{min(unique_nodes)}",
        f"{max(unique_nodes)}",
        f"{sorted(unique_nodes)[0:500]}",
    )
    return edges, aligned_labels, unique_nodes[valid_mask]


def _align_nodes(heads, tails, ids, labels):
    """Map node IDs to contiguous indices and align edges and labels."""
    unique_nodes, inverse_idx = np.unique(np.concatenate([heads, tails]), return_inverse=True)
    edges = inverse_idx.reshape(-1, 2)
    id_to_label = dict(zip(ids, labels))
    aligned_labels = np.array([id_to_label.get(nid, -1) for nid in unique_nodes], dtype=np.int64)
    valid_mask = aligned_labels != -1
    edges = edges[np.all(valid_mask[edges], axis=1)]
    aligned_labels = aligned_labels[valid_mask]
    return edges, aligned_labels


def _compute_acc(true_labels, pred_labels):
    """Compute clustering accuracy via majority label assignment."""
    acc = 0
    for c in np.unique(pred_labels):
        idx = np.where(pred_labels == c)[0]
        if len(idx) == 0:
            continue
        majority_label = mode(true_labels[idx], keepdims=False).mode
        acc += np.sum(true_labels[idx] == majority_label)
    return acc / len(true_labels)


def compute_metrics(true_labels, pred_labels):
    """Compute clustering evaluation metrics."""
    return {
        "NMI": normalized_mutual_info_score(true_labels, pred_labels),
        "ARI": adjusted_rand_score(true_labels, pred_labels),
        "ACC": cluster_accuracy(true_labels, pred_labels),
    }


def cluster_accuracy(true_labels, pred_labels):
    """Simple cluster accuracy via majority voting."""

    acc = 0
    for c in np.unique(pred_labels):
        idx = np.where(pred_labels == c)[0]
        majority_label = mode(true_labels[idx], keepdims=False).mode
        acc += np.sum(true_labels[idx] == majority_label)
    return acc / len(true_labels)


def _compute_community_homophily(true_labels, comm_labels):
    """
    Compute homophily for each community based on ground truth labels.
    Homophily of a community = ratio of majority label nodes to total nodes in that community.
    Returns: array of homophily values, one per community.
    """
    homophilies = []
    for c in np.unique(comm_labels):
        idx = np.where(comm_labels == c)[0]
        labels_in_comm = true_labels[idx]
        if len(labels_in_comm) == 0:
            homophilies.append(0.0)
        else:
            majority_ratio = np.max(np.bincount(labels_in_comm)) / len(labels_in_comm)
            homophilies.append(majority_ratio)
    return np.array(homophilies)


def louvain(edges: List[Tuple]):
    partition = community_louvain.best_partition(Graph(edges))
    return partition


def leiden(edges: List[Tuple]):
    g = ig.Graph.TupleList(edges)
    partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
    return partition


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
    same = u_labels == v_labels

    return same.sum() / len(edges) if len(edges) > 0 else 0.0


def get_labels(
    task_tab_path="./data/outbrain/autog/ctr/final/data/Click.pqt",
    element_col="cl_ad_id",
    label_col="clicked",
):
    label_tab = pd.read_parquet(task_tab_path, columns=[element_col, label_col])
    return label_tab[element_col].values, label_tab[label_col].values


def get_edges(
    task_tab_path="./data/outbrain/autog/ctr/final/data/Ad_Advertiser_Ad.pqt",
    head_col="head_ad_id",
    tail_col="tail_ad_id",
):
    label_tab = pd.read_parquet(task_tab_path, columns=[head_col, tail_col])
    return label_tab[head_col].values, label_tab[tail_col].values
