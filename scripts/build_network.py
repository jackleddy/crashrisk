import json
from pathlib import Path

import geopandas as gpd
import osmnx as ox

from crashrisk.config import Outputs, REGION_BUFFER, REGION_NAME
from crashrisk.ingest.regions import get_region

def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def _serialize_for_parquet(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return json.dumps(list(value), sort_keys=True)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, default=str, sort_keys=True)

def _normalize_object_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    normalized = gdf.copy()
    for column in normalized.columns:
        series = normalized[column]
        if series.dtype != object:
            continue

        non_null = series.dropna()
        if non_null.empty:
            continue

        types = {type(value) for value in non_null}
        has_nested = any(isinstance(value, (list, tuple, set, dict)) for value in non_null)
        if has_nested or len(types) > 1:
            normalized[column] = series.map(_serialize_for_parquet)

    return normalized

def main() -> None:
    out = Outputs()

    print(f"Region: {REGION_NAME}")
    poly = get_region(REGION_NAME, buffer_m=REGION_BUFFER)

    ox.settings.use_cache = True
    ox.settings.log_console = True

    print("Downloading OSM drive network...")
    G = ox.graph_from_polygon(poly, network_type="drive", simplify=True)

    print("Projecting graph to UTM ...")
    G = ox.project_graph(G)

    print("Exporting graph and tables ...")
    ensure_parent(out.osm_graph_file)
    ox.save_graphml(G, out.osm_graph_file)

    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
    nodes = _normalize_object_columns(nodes)
    edges = _normalize_object_columns(edges)

    ensure_parent(out.osm_nodes_file)
    ensure_parent(out.osm_edges_file)
    nodes.reset_index().to_parquet(out.osm_nodes_file, index=False)
    edges.reset_index().to_parquet(out.osm_edges_file, index=False)

    print(f"Wrote: {out.osm_graph_file}")
    print(f"Wrote: {out.osm_nodes_file}")
    print(f"Wrote: {out.osm_edges_file}")
    print(f"Nodes: {len(nodes):,} | Edges: {len(edges):,}")


if __name__ == "__main__":
    main()
