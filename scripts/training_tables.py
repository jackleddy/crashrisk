from pathlib import Path

import geopandas as gpd
import pandas as pd

from crashrisk.config import (
    Outputs,
    CRASH_CONTEXT_FEATURES,
    EDGE_TRAIN_FEATURES,
    NODE_TRAIN_FEATURES,
)
from crashrisk.labeling.build_training_tables import (
    build_edge_training_table,
    build_node_training_table,
)


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    out = Outputs()

    # Inputs
    edges_with_traffic = gpd.read_parquet(out.edges_with_traffic_file)
    nodes = gpd.read_parquet(out.osm_nodes_file)
    crashes = gpd.read_parquet(out.crashes_snapped_file)

    if edges_with_traffic.crs is not None and nodes.crs is not None and str(edges_with_traffic.crs) != str(nodes.crs):
        nodes = nodes.to_crs(edges_with_traffic.crs)

    train_edges = build_edge_training_table(
        edges_with_traffic=edges_with_traffic,
        crashes_snapped=crashes,
        crash_context_cols=CRASH_CONTEXT_FEATURES,
        edge_feature_cols=EDGE_TRAIN_FEATURES,
    )
    ensure_parent(out.train_edges_file)
    train_edges.to_parquet(out.train_edges_file, index=False)
    print(f"Wrote: {out.train_edges_file}  ({len(train_edges):,} rows)")

    train_nodes = build_node_training_table(
        nodes=nodes,
        edges_with_traffic=edges_with_traffic,
        crashes_snapped=crashes,
        crash_context_cols=CRASH_CONTEXT_FEATURES,
        node_feature_cols=NODE_TRAIN_FEATURES,
    )
    ensure_parent(out.train_nodes_file)
    train_nodes.to_parquet(out.train_nodes_file, index=False)
    print(f"Wrote: {out.train_nodes_file}  ({len(train_nodes):,} rows)")

    print("\nQA:")
    print("Edges: y>0:", int((train_edges["y"] > 0).sum()), "/", len(train_edges))
    print("Nodes: y>0:", int((train_nodes["y"] > 0).sum()), "/", len(train_nodes))
    print("Edge exposure summary:\n", train_edges["exposure"].describe())
    print("Node exposure summary:\n", train_nodes["exposure"].describe())


if __name__ == "__main__":
    main()