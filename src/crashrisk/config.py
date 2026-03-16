from dataclasses import dataclass
from typing import Sequence


REGION_NAME: str = "Blacksburg, Virginia, USA"
REGION_BUFFER: float = 300.0
CRASH_YEAR_MIN: int = 2018
CRASH_YEAR_MAX: int = 2026
# max api allows in one pull is 16000 + 2000, so will have to recursively pull for larger sets
CRASH_PAGE_SIZE: int = 16000    
TRAFFIC_PAGE_SIZE: int = 2000

CRASH_LAYER_URL: str = (
    "https://services.arcgis.com/p5v98VHDX9Atv3l7/ArcGIS/rest/services/Full_Crash/FeatureServer/0"
)
TRAFFIC_VOLUME_LAYER_URL: str = (
    "https://services.arcgis.com/p5v98VHDX9Atv3l7/arcgis/rest/services/VDOT_Traffic_Volume_2024/FeatureServer/0"
)

# Fields we pull
CRASH_OUTFIELDS: Sequence[str] = (
    "DOCUMENT_NBR",
    "CRASH_YEAR",
    "CRASH_DT",
    "CRASH_MILITARY_TM",
    "CRASH_SEVERITY",
    "WEATHER_CONDITION",
    "LIGHT_CONDITION",
    "ROADWAY_SURFACE_COND",
    "ROADWAY_ALIGNMENT",
    "INTERSECTION_TYPE",
    "TRAFFIC_CONTROL_TYPE",
    "TRFC_CTRL_STATUS_TYPE",
    "RELATION_TO_ROADWAY",
    "ROADWAY_SURFACE_TYPE",
    "ROADWAY_DESCRIPTION",
)
TRAFFIC_OUTFIELDS: Sequence[str] = (
    "EVENT_SOURCE_ID",
    "DATA_DATE",
    "ROUTE_COMMON_NAME",
    "ROUTE_NAME",
    "RTE_TYPE_CD",
    "ADT",
    "ADT_QUALITY",
    "AAWDT",
)

# Training table features
CRASH_CONTEXT_FEATURES: Sequence[str] = (
    "WEATHER_CONDITION",
    "LIGHT_CONDITION",
    "ROADWAY_SURFACE_COND",
    "ROADWAY_ALIGNMENT",
    "INTERSECTION_TYPE",
    "TRAFFIC_CONTROL_TYPE",
)
EDGE_TRAIN_FEATURES: Sequence[str] = (
    "length", 
    "highway",          # road class
    "oneway",
    "lanes",
    "maxspeed",
    "ADT",              # from traffic volume 
    "AAWDT", 
    "tv_match_dist_m",  # match quality signal
    "curvature",        # computed
    "adt_used",         # computed
)
NODE_TRAIN_FEATURES: Sequence[str] = (
    "degree",
    "x",
    "y",
    "adt_sum_incident", # computed
)

@dataclass(frozen=True)
class Outputs:
    """Output locations for intermediate and processed data."""

    raw_dir: str = "data/raw"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"
    
    # Change names if location changes
    crashes_file: str = "data/raw/blacksburg_crashes.parquet"
    traffic_file: str = "data/raw/blacksburg_traffic_volume.parquet"

    osm_graph_file: str = "data/processed/osm_drive_graph.graphml"
    osm_nodes_file: str = "data/processed/osm_nodes.parquet"
    osm_edges_file: str = "data/processed/osm_edges.parquet"

    edges_with_traffic_file: str = "data/interim/osm_edges_with_traffic.parquet"
    crashes_snapped_file: str = "data/interim/crashes_snapped.parquet"

    train_edges_file: str = "data/processed/train_edges.parquet"
    train_nodes_file: str = "data/processed/train_nodes.parquet"

    # GNN
    gnn_model_file: str = "data/processed/gnn_graphsage_edge_poisson.pt"
    gnn_edge_predictions_file: str = "data/processed/gnn_edge_predictions.parquet"

    # Map output
    risk_map_html_file: str = "data/processed/risk_map.html"


# GNN CONFIG

HIGHWAY_CATEGORIES = [
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "residential", "service", "unclassified", "living_street",
]

EDGE_NUMERIC_FEATURES = [
    "curvature",   # >=1.0
    "lanes_num",
    "maxspeed_num",
    "oneway_num",
]

CURVATURE_MAX_CLIP: float = 20.0
EXPOSURE_MIN_CLIP: float = 1e-6
EXPOSURE_MAX_CLIP: float = 1e8

NODE_NUMERIC_FEATURES = [
    "in_deg",
    "out_deg",
    "deg",
]
