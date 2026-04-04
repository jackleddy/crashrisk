from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import torch
from torch_geometric.data import Data

from crashrisk.config import (
    CURVATURE_MAX_CLIP,
    EDGE_CATEGORICAL_FEATURES,
    EDGE_NUMERIC_FEATURES,
    EXPOSURE_MAX_CLIP,
    EXPOSURE_MIN_CLIP,
    HIGHWAY_CATEGORIES,
    NODE_NUMERIC_FEATURES,
)


def _pick_node_id_col(nodes: pd.DataFrame) -> str:
    for c in ("node_id", "osmid", "id"):
        if c in nodes.columns:
            return c
    raise ValueError("Could not find node id column in OSM nodes (expected 'osmid' or 'node_id').")


def _to_float(x):
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return np.nan

    # parse first number in string
    num = ""
    for ch in s:
        if ch.isdigit() or ch == ".":
            num += ch
        elif num:
            break
    try:
        return float(num) if num else np.nan
    except Exception:
        return np.nan


def _standardize(mat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = np.nanmean(mat, axis=0)
    sd = np.nanstd(mat, axis=0)
    sd = np.where(sd < eps, 1.0, sd)
    return (np.nan_to_num(mat, nan=0.0) - mu) / sd

def _make_edge_splits(
        y: np.ndarray, seed: int = 42, train = 0.7, val = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx_pos = np.where(y > 0)[0]
    idx_neg = np.where(y == 0)[0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)

    def split_idx(idx):
        n = len(idx)
        n_tr = int(train * n)
        n_va = int(val * n)
        tr = idx[:n_tr]
        va = idx[n_tr:n_tr+n_va]
        te = idx[n_tr+n_va:]
        return tr, va, te

    trp, vap, tep = split_idx(idx_pos)
    trn, van, ten = split_idx(idx_neg)

    train_idx = np.concatenate([trp, trn])
    val_idx   = np.concatenate([vap, van])
    test_idx  = np.concatenate([tep, ten])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return train_idx, val_idx, test_idx


@dataclass
class EdgePredictionBatch:
    data: Data                       # graph for message passing (x, edge_index)
    edge_u: torch.Tensor             # [M] source node indices for prediction edges
    edge_v: torch.Tensor             # [M] target node indices for prediction edges
    edge_attr: torch.Tensor          # [M, F_edge] edge features
    y: torch.Tensor                  # [M] crash counts
    log_exposure: torch.Tensor       # [M] log(exposure)
    edge_ids: List[str]              # [M] edge_id strings
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    test_idx: torch.Tensor


def build_edge_dataset(
    nodes_path: str,
    edges_with_traffic_path: str,
    train_edges_path: str,
    device: str = "cpu",
) -> EdgePredictionBatch:
    nodes = gpd.read_parquet(nodes_path)
    edges = gpd.read_parquet(edges_with_traffic_path)
    train_edges = pd.read_parquet(train_edges_path)

    node_id_col = _pick_node_id_col(nodes)
    node_ids = nodes[node_id_col].astype(int).to_numpy()
    node_to_idx: Dict[int, int] = {int(nid): i for i, nid in enumerate(node_ids)}
    N = len(node_ids)

    if not all(c in edges.columns for c in ("u", "v", "edge_id")):
        raise ValueError("edges_with_traffic must contain u, v, edge_id columns.")

    # Build edge_index for message passing
    u_idx = edges["u"].astype(int).map(node_to_idx).to_numpy()
    v_idx = edges["v"].astype(int).map(node_to_idx).to_numpy()
    if np.any(pd.isna(u_idx)) or np.any(pd.isna(v_idx)):
        raise ValueError("Some edge endpoints not found in node table (node_id mapping mismatch).")

    u_idx = u_idx.astype(np.int64)
    v_idx = v_idx.astype(np.int64)
    edge_index = np.stack([u_idx, v_idx], axis=0)
    edge_index_rev = np.stack([v_idx, u_idx], axis=0)
    edge_index_undirected = np.concatenate([edge_index, edge_index_rev], axis=1)
    edge_index_t = torch.tensor(edge_index_undirected, dtype=torch.long, device=device)

    # Node features: degrees from directed edges
    in_deg = np.zeros(N, dtype=np.float32)
    out_deg = np.zeros(N, dtype=np.float32)
    for a, b in zip(u_idx, v_idx):
        out_deg[a] += 1.0
        in_deg[b] += 1.0
    deg = in_deg + out_deg
    X_node = np.stack([in_deg, out_deg, deg], axis=1)
    X_node = _standardize(X_node).astype(np.float32)
    x_t = torch.tensor(X_node, dtype=torch.float, device=device)

    data = Data(x=x_t, edge_index=edge_index_t)

    # Prediction edges and labels

    if "edge_id" not in train_edges.columns:
        raise ValueError("train_edges.parquet must contain edge_id.")

    label_cols = ["edge_id", "y", "exposure", *EDGE_CATEGORICAL_FEATURES]
    if "curvature" in train_edges.columns:
        label_cols.append("curvature")

    train_labels = train_edges[label_cols].drop_duplicates(
        subset=["edge_id"], keep="first"
    )
    merged = edges.merge(train_labels, on="edge_id", how="left")
    merged["y"] = merged["y"].fillna(0).astype(int)
    merged["exposure"] = (
        merged["exposure"]
        .fillna(1.0)
        .astype(float)
        .clip(lower=EXPOSURE_MIN_CLIP, upper=EXPOSURE_MAX_CLIP)
    )

    lanes_src = merged["lanes"] if "lanes" in merged.columns else pd.Series(np.nan, index=merged.index)
    maxspeed_src = (
        merged["maxspeed"] if "maxspeed" in merged.columns else pd.Series(np.nan, index=merged.index)
    )
    merged["lanes_num"] = lanes_src.map(_to_float)
    merged["maxspeed_num"] = maxspeed_src.map(_to_float)
    oneway_src = merged["oneway"] if "oneway" in merged.columns else pd.Series(False, index=merged.index)
    merged["oneway_num"] = oneway_src.map(
        lambda v: 1.0 if str(v).lower() in ("true", "1", "yes") else 0.0
    )
    
    if "curvature" not in merged.columns:
        merged["curvature"] = 1.0
    merged["curvature"] = (
        merged["curvature"]
        .map(_to_float)
        .fillna(1.0)
        .astype(float)
        .clip(lower=1.0, upper=CURVATURE_MAX_CLIP)
    )

    # Build numeric matrix
    X_edge_num = np.stack([merged[c].map(_to_float).to_numpy() for c in EDGE_NUMERIC_FEATURES], axis=1).astype(np.float32)
    X_edge_num = _standardize(X_edge_num).astype(np.float32)

    categorical_cols = []
    for col in EDGE_CATEGORICAL_FEATURES:
        if col in merged.columns:
            col_ohe = pd.get_dummies(merged[col].fillna("missing").astype(str), prefix=col)
            categorical_cols.append(col_ohe)

    X_edge = np.concatenate([X_edge_num, *categorical_cols], axis=1).astype(np.float32)
    edge_attr_t = torch.tensor(X_edge, dtype=torch.float, device=device)

    edge_u_t = torch.tensor(u_idx, dtype=torch.long, device=device)
    edge_v_t = torch.tensor(v_idx, dtype=torch.long, device=device)

    y_t = torch.tensor(merged["y"].to_numpy(dtype=np.float32), dtype=torch.float, device=device)
    log_exp_t = torch.tensor(np.log(merged["exposure"].to_numpy(dtype=np.float32)), dtype=torch.float, device=device)

    edge_ids = merged["edge_id"].astype(str).tolist()

    train_i, val_i, test_i = _make_edge_splits(merged["y"].to_numpy(), seed=42)
    train_idx_t = torch.tensor(train_i, dtype=torch.long, device=device)
    val_idx_t = torch.tensor(val_i, dtype=torch.long, device=device)
    test_idx_t = torch.tensor(test_i, dtype=torch.long, device=device)

    return EdgePredictionBatch(
        data=data,
        edge_u=edge_u_t,
        edge_v=edge_v_t,
        edge_attr=edge_attr_t,
        y=y_t,
        log_exposure=log_exp_t,
        edge_ids=edge_ids,
        train_idx=train_idx_t,
        val_idx=val_idx_t,
        test_idx=test_idx_t,
    )
