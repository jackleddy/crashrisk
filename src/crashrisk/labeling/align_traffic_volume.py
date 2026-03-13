from dataclasses import dataclass
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd

from crashrisk.labeling.spatial_match import (
    make_edge_id,
    midpoint_of_linestring,
    nearest_join,
    bearing_deg,
    ang_diff_deg,
)
from crashrisk.config import TRAFFIC_OUTFIELDS

# Alignment matching parameters
@dataclass(frozen=True)
class TrafficAlignConfig:
    max_distance_m: float = 60.0        # tune 
    max_bearing_diff_deg: float = 45.0  # could include this but idk may be annoying


def align_traffic_volume_to_osm_edges(
    osm_edges: gpd.GeoDataFrame,
    traffic_segments: gpd.GeoDataFrame,
    cfg: TrafficAlignConfig = TrafficAlignConfig(),
) -> gpd.GeoDataFrame:
    if osm_edges.crs is None or traffic_segments.crs is None:
        raise ValueError("Both osm_edges and traffic_segments must have CRS")
    
    for col in ("u", "v", "key", "geometry"):
        if col not in osm_edges.columns:
            raise ValueError(f"osm_edges is missing required column '{col}'")

    edges = osm_edges.copy()
    edges["edge_id"] = edges.apply(lambda r: make_edge_id(r["u"], r["v"], r["key"]), axis=1)
    midpoints = gpd.GeoDataFrame(
        {"edge_id": edges["edge_id"].values},
        geometry=edges.geometry.apply(midpoint_of_linestring),
        crs=edges.crs,
    )

    traffic_keep = [c for c in TRAFFIC_OUTFIELDS if c in traffic_segments.columns]

    matched = nearest_join(
        midpoints,
        traffic_segments,
        right_keep_cols=traffic_keep,
        max_distance_m=cfg.max_distance_m,
        distance_col="tv_match_dist_m",
    )
    
    out = edges.merge(
        matched.drop(columns=["geometry"], errors="ignore"),
        on="edge_id",
        how="left",
        suffixes=("", "_tv"),
    )

    return out