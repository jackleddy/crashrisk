from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from crashrisk.config import CURVATURE_MAX_CLIP, EXPOSURE_MAX_CLIP, EXPOSURE_MIN_CLIP

# Helpers

def _pick_node_id_col(nodes: pd.DataFrame) -> str:
    for c in ("node_id", "osmid", "id"):
        if c in nodes.columns:
            return c
    raise ValueError("Could not find node id column in OSM nodes (expected 'osmid' or 'node_id').")


def _to_numeric_safe(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    
    num = ""
    for ch in s:
        if ch.isdigit() or ch == ".":
            num += ch
        elif num:
            break
    try:
        return float(num) if num else None
    except Exception:
        return None


def _mode_or_nan(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return np.nan

    m = s.mode()
    return m.iloc[0] if not m.empty else np.nan


def compute_curvature(lines: LineString, length_m: Optional[float]) -> float:
    if lines is None:
        return 1.0
    coords = list(lines.coords)
    if len(coords) < 2:
        return 1.0
    x1, y1 = coords[0]
    x2, y2 = coords[-1]
    chord = float(np.hypot(x2 - x1, y2 - y1))
    L = float(length_m) if length_m is not None else float(lines.length)
    eps = 1e-6

    curvature = float(L / max(chord, eps))
    return float(min(curvature, CURVATURE_MAX_CLIP)) # path length / straight endpoint distance


# Aggregate 

def aggregate_crashes(
        crashes: pd.DataFrame,
        id_col: str,
        context_cols: Sequence[str],
) -> pd.DataFrame:
    df = crashes.copy()
    df = df[df[id_col].notna()]

    agg_dict = {c: _mode_or_nan for c in context_cols}
    grouped = df.groupby(id_col, dropna=True).agg(agg_dict)
    grouped = grouped.rename(columns={c: f"{c}_mode" for c in context_cols})

    counts = df.groupby(id_col, dropna=True).size().rename("y")
    out = pd.concat([counts, grouped], axis=1).reset_index()
    return out

# Exposure for crash rate approx

def compute_edge_adt_used(edges: pd.DataFrame) -> pd.Series:
    adt = edges.get("ADT", pd.Series([np.nan] * len(edges)))
    aawdt = edges.get("AAWDT", pd.Series([np.nan] * len(edges)))

    adt_num = adt.map(_to_numeric_safe)
    aawdt_num = aawdt.map(_to_numeric_safe)

    used = adt_num.copy()
    used = used.where(pd.notna(used), aawdt_num)
    used = used.fillna(1.0)
    # Avoid 0 exposure
    used = used.where(used > 0, 1.0)
    return used.astype(float)


def compute_edge_exposure(edges: pd.DataFrame, adt_used: pd.Series) -> pd.Series:
    if "length" not in edges.columns:
        raise ValueError("Edges must have 'length' column (meters).")
    length = edges["length"].map(_to_numeric_safe).fillna(edges["length"]).astype(float)
    exp = adt_used.astype(float) * length
    exp = exp.clip(lower=EXPOSURE_MIN_CLIP, upper=EXPOSURE_MAX_CLIP)
    return exp


def compute_node_exposure_from_incident_edges(
    nodes: pd.DataFrame,
    node_id_col: str,
    edges: pd.DataFrame,
    adt_used: pd.Series,
) -> pd.DataFrame:
    if "u" not in edges.columns or "v" not in edges.columns:
        raise ValueError("Edges must have 'u' and 'v' columns for incidence.")

    e = edges[["u", "v"]].copy()
    e["adt_used"] = adt_used.values

    sum_u = e.groupby("u")["adt_used"].sum()
    sum_v = e.groupby("v")["adt_used"].sum()

    # align to nodes
    node_ids = nodes[node_id_col]
    adt_sum = node_ids.map(sum_u).fillna(0.0) + node_ids.map(sum_v).fillna(0.0)

    adt_sum = adt_sum.where(adt_sum > 0, 1.0)

    return pd.DataFrame({node_id_col: node_ids.values, "adt_sum_incident": adt_sum.astype(float).values})


# Training table

def build_edge_training_table(
    edges_with_traffic: gpd.GeoDataFrame,
    crashes_snapped: pd.DataFrame,
    crash_context_cols: Sequence[str],
    edge_feature_cols: Sequence[str],
) -> pd.DataFrame:
    """
    Output schema:
      edge_id, y, exposure, rate,
      {context}_mode...,
      selected edge feature cols...
    """
    edges = edges_with_traffic.copy()

    # Ensure edge_id exists
    if "edge_id" not in edges.columns:
        if not all(c in edges.columns for c in ("u", "v", "key")):
            raise ValueError("edges_with_traffic must have edge_id or u/v/key columns.")
        edges["edge_id"] = edges.apply(lambda r: f"{int(r['u'])}_{int(r['v'])}_{int(r['key'])}", axis=1)

    edge_labels = aggregate_crashes(crashes_snapped, id_col="edge_id", context_cols=crash_context_cols)

    out = edges.merge(edge_labels, on="edge_id", how="left")
    out["y"] = out["y"].fillna(0).astype(int)

    out["curvature"] = out.apply(
        lambda r: compute_curvature(r["geometry"], _to_numeric_safe(r.get("length"))),
        axis=1,
    )
    out["adt_used"] = compute_edge_adt_used(out)
    out["exposure"] = compute_edge_exposure(out, out["adt_used"])
    out["rate"] = out["y"].astype(float) / out["exposure"].astype(float)

    keep = ["edge_id", "y", "exposure", "rate"] + [f"{c}_mode" for c in crash_context_cols]
    for c in edge_feature_cols:
        if c not in keep and c in out.columns:
            keep.append(c)

    # Optionally keep centroid coords
    cent = out.geometry.centroid
    out["x_centroid"] = cent.x
    out["y_centroid"] = cent.y
    keep += ["x_centroid", "y_centroid"]

    final = out[keep].copy()
    return final


def build_node_training_table(
    nodes: gpd.GeoDataFrame,
    edges_with_traffic: pd.DataFrame,
    crashes_snapped: pd.DataFrame,
    crash_context_cols: Sequence[str],
    node_feature_cols: Sequence[str],
) -> pd.DataFrame:
    """
    Output schema:
      node_id, y, exposure, rate,
      {context}_mode...,
      selected node feature cols...
    """
    n = nodes.copy()
    node_id_col = _pick_node_id_col(n)

    node_labels = aggregate_crashes(crashes_snapped, id_col="node_id", context_cols=crash_context_cols)

    out = n.merge(node_labels, left_on=node_id_col, right_on="node_id", how="left", suffixes=("", "_label"))
    y_label_col = "y_label" if "y_label" in out.columns else "y"
    out["y"] = out[y_label_col].fillna(0).astype(int)

    # Ensure degree exists (compute from edges if missing)
    if "degree" not in out.columns:
        if not all(c in edges_with_traffic.columns for c in ("u", "v")):
            raise ValueError("Need edges u/v to compute node degree.")
        deg_u = edges_with_traffic.groupby("u").size()
        deg_v = edges_with_traffic.groupby("v").size()
        out["degree"] = out[node_id_col].map(deg_u).fillna(0) + out[node_id_col].map(deg_v).fillna(0)

    e = edges_with_traffic.copy()
    if "adt_used" not in e.columns:
        e["adt_used"] = compute_edge_adt_used(e)

    node_exp = compute_node_exposure_from_incident_edges(out, node_id_col, e, e["adt_used"])
    out = out.merge(node_exp, on=node_id_col, how="left")
    out["adt_sum_incident"] = out["adt_sum_incident"].fillna(1.0)

    out["exposure"] = out["adt_sum_incident"].astype(float)  # simple proxy
    out["exposure"] = out["exposure"].where(out["exposure"] > 0, 1e-6)
    out["rate"] = out["y"].astype(float) / out["exposure"].astype(float)

    # Keep limited fields
    keep = [node_id_col, "y", "exposure", "rate"] + [f"{c}_mode" for c in crash_context_cols]
    for c in node_feature_cols:
        if c not in keep and c in out.columns:
            keep.append(c)

    # Normalize id column name to "node_id" for downstream consistency
    final = out[keep].copy().rename(columns={node_id_col: "node_id"})

    return final
