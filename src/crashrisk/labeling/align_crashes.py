from dataclasses import dataclass

import geopandas as gpd
import numpy as np

from crashrisk.labeling.spatial_match import (
    nearest_join,
    is_intersection_crash,
)


@dataclass(frozen=True)
class CrashSnapConfig:
    edge_max_distance_m: float = 60.0
    node_max_distance_m: float = 25.0


def snap_crashes_to_network(
    crashes: gpd.GeoDataFrame,
    osm_edges: gpd.GeoDataFrame,
    osm_nodes: gpd.GeoDataFrame,
    cfg: CrashSnapConfig = CrashSnapConfig(),
) -> gpd.GeoDataFrame:
    if crashes.crs is None or osm_edges.crs is None or osm_nodes.crs is None:
        raise ValueError("crashes, osm_edges, and osm_nodes must all have CRS set.")

    for col in ("u", "v", "key", "geometry"):
        if col not in osm_edges.columns:
            raise ValueError(f"osm_edges missing required '{col}'")

    node_id_col = None
    for candidate in ("node_id", "osmid", "id"):
        if candidate in osm_nodes.columns:
            node_id_col = candidate
            break
    if node_id_col is None:
        raise ValueError("osm_nodes must contain a node id column (e.g., 'osmid' or 'node_id').")

    edges = osm_edges.copy()
    edges["edge_id"] = (
        edges["u"].map(int).astype(str)
        + "_"
        + edges["v"].map(int).astype(str)
        + "_"
        + edges["key"].map(int).astype(str)
    )

    # Nearest edge for all crashes
    edge_keep = ["edge_id"]
    crashes_edge = nearest_join(
        crashes,
        edges[edge_keep + ["geometry"]],
        right_keep_cols=edge_keep,
        max_distance_m=cfg.edge_max_distance_m,
        distance_col="edge_snap_dist_m",
    )
    crashes_edge = crashes_edge.reset_index(drop=True)
    if "RELATION_TO_ROADWAY" in crashes_edge.columns:
        crashes_edge["intersection_flag"] = crashes_edge["RELATION_TO_ROADWAY"].apply(
            is_intersection_crash
        )
    else:
        crashes_edge["intersection_flag"] = False

    # Nearest node for all crashes
    crashes_both = crashes_edge.copy()
    crashes_both["node_id"] = np.nan
    crashes_both["node_snap_dist_m"] = np.nan
    intersection_idx = crashes_both.index[crashes_both["intersection_flag"]]
    if len(intersection_idx):
        node_keep_col = "__snap_node_id"
        nodes_for_join = osm_nodes[[node_id_col, "geometry"]].rename(
            columns={node_id_col: node_keep_col}
        )
        crashes_node = nearest_join(
            crashes_both.loc[intersection_idx],
            nodes_for_join,
            right_keep_cols=[node_keep_col],
            max_distance_m=cfg.node_max_distance_m,
            distance_col="node_snap_dist_m",
        )
        # Handle equidistant ties from sjoin_nearest: keep one node per crash index.
        crashes_node = crashes_node.sort_values("node_snap_dist_m")
        crashes_node = crashes_node[~crashes_node.index.duplicated(keep="first")]

        node_id_aligned = crashes_node[node_keep_col].reindex(intersection_idx)
        node_dist_aligned = crashes_node["node_snap_dist_m"].reindex(intersection_idx)
        crashes_both.loc[intersection_idx, "node_id"] = node_id_aligned
        crashes_both.loc[intersection_idx, "node_snap_dist_m"] = node_dist_aligned

    # If crash is at an intersection assign it to a node
    has_edge = crashes_both["edge_id"].notna()
    has_node = crashes_both["node_id"].notna()

    assigned = np.full(len(crashes_both), "none", dtype=object)
    assigned[(crashes_both["intersection_flag"]) & has_node] = "node"
    assigned[(crashes_both["intersection_flag"]) & (~has_node) & has_edge] = "edge"
    assigned[(~crashes_both["intersection_flag"]) & has_edge] = "edge"

    crashes_both["assigned_to"] = assigned

    return crashes_both
