from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from math import atan2, degrees

# Helpers
def make_edge_id(u: int, v: int, key: int) -> str:
    return f"{int(u)}_{int(v)}_{int(key)}"

def ensure_crs(gdf: gpd.GeoDataFrame, crs: str | int) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        return gdf.set_crs(crs)
    return gdf

def project_to(gdf: gpd.GeoDataFrame, target_crs) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS. Cannot reproject safely.")
    if str(gdf.crs) != str(target_crs):
        return gdf.to_crs(target_crs)
    return gdf

def midpoint_of_linestring(ls: LineString) -> Point:
    return ls.interpolate(0.5, normalized=True)

def bearing_deg(linestring: LineString) -> float:
    coords = list(linestring.coords)
    if len(coords) < 2:
        return 0.0
    x1, y1 = coords[0]
    x2, y2 = coords[-1]
    dx = x2 - x1
    dy = y2 - y1
    # compass bearing (north is 0, east is 90)
    b = (degrees(atan2(dx, dy)) + 360.0) % 360.0
    return float(b)

def ang_diff_deg(a: float, b: float) -> float:
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


# Join logic

def nearest_join(
        left: gpd.GeoDataFrame,
        right: gpd.GeoDataFrame,
        right_keep_cols: Sequence[str],
        *,
        max_distance_m: Optional[float] = None,
        distance_col: str = "match_dist_m",
) -> gpd.GeoDataFrame:
    if left.crs is None or right.crs is None:
        raise ValueError("Both GeoDataFrames must have CRS set")
    
    right_small = right[list(set(right_keep_cols) | {"geometry"})].copy()

    joined = gpd.sjoin_nearest(
        left,
        right_small,
        how="left",
        max_distance=max_distance_m,
        distance_col=distance_col,
    )

    if "index_right" in joined.columns:
        joined = joined.drop(columns=["index_right"])

    return joined


def find_closest_edge(
    point: Point,
    edges: gpd.GeoDataFrame,
    *,
    max_distance_m: Optional[float] = None,
) -> tuple[Optional[pd.Series], Optional[float]]:
    g = gpd.GeoDataFrame({"geometry": [point]}, crs=edges.crs)
    j = nearest_join(g, edges, right_keep_cols=list(edges.columns), max_distance_m=max_distance_m)

    if j.empty or pd.isna(j.iloc[0].get("match_dist_m", np.nan)):
        return None, None
    dist = float(j.iloc[0]["match_dist_m"])
    edge_row = j.iloc[0][edges.columns]

    return edge_row, dist


def find_closest_node(
    point: Point,
    nodes: gpd.GeoDataFrame,
    *,
    max_distance_m: Optional[float] = None,
) -> tuple[Optional[pd.Series], Optional[float]]:
    g = gpd.GeoDataFrame({"geometry": [point]}, crs=nodes.crs)
    j = nearest_join(g, nodes, right_keep_cols=list(nodes.columns), max_distance_m=max_distance_m)

    if j.empty or pd.isna(j.iloc[0].get("match_dist_m", np.nan)):
        return None, None
    dist = float(j.iloc[0]["match_dist_m"])
    node_row = j.iloc[0][nodes.columns]

    return node_row, dist


def is_intersection_crash(relation_to_roadway) -> bool:
    if relation_to_roadway is None:
        return False
    s = str(relation_to_roadway).strip().lower()
    if "9" in s:
        return True
    return False