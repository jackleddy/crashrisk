from pathlib import Path

import geopandas as gpd

from crashrisk.config import Outputs
from crashrisk.labeling.align_traffic_volume import align_traffic_volume_to_osm_edges, TrafficAlignConfig
from crashrisk.labeling.align_crashes import snap_crashes_to_network, CrashSnapConfig


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    out = Outputs()

    crashes = gpd.read_parquet(out.crashes_file)
    traffic = gpd.read_parquet(out.traffic_file)
    osm_edges = gpd.read_parquet(out.osm_edges_file)
    osm_nodes = gpd.read_parquet(out.osm_nodes_file)

    target_crs = osm_edges.crs
    if target_crs is None:
        raise ValueError("OSM edges have no CRS. Rebuild OSM graph and ensure CRS is saved.")

    crashes = crashes.to_crs(target_crs)
    traffic = traffic.to_crs(target_crs)
    osm_nodes = osm_nodes.to_crs(target_crs)

    aligned_edges = align_traffic_volume_to_osm_edges(
        osm_edges=osm_edges,
        traffic_segments=traffic,
        cfg=TrafficAlignConfig(max_distance_m=60.0, max_bearing_diff_deg=None),
    )

    edges_out = "data/interim/osm_edges_with_traffic.parquet"
    ensure_parent(edges_out)
    aligned_edges.to_parquet(edges_out, index=False)
    print(f"Wrote: {edges_out}  ({len(aligned_edges):,} edges)")

    snapped = snap_crashes_to_network(
        crashes=crashes,
        osm_edges=osm_edges,
        osm_nodes=osm_nodes,
        cfg=CrashSnapConfig(edge_max_distance_m=60.0, node_max_distance_m=25.0),
    )

    crashes_out = "data/interim/crashes_snapped.parquet"
    ensure_parent(crashes_out)
    snapped.to_parquet(crashes_out, index=False)
    print(f"Wrote: {crashes_out}  ({len(snapped):,} crashes)")

    print("\nQA:")
    print("  Edge matched:", int(snapped["edge_id"].notna().sum()), "/", len(snapped))
    print("  Node matched:", int(snapped["node_id"].notna().sum()), "/", len(snapped))
    print("  Assigned_to counts:\n", snapped["assigned_to"].value_counts(dropna=False))


if __name__ == "__main__":
    main()