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


@dataclass(frozen=True)
class Outputs:
    """Output locations for intermediate and processed data."""

    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    
    # Change names 
    crashes_file: str = "data/raw/blacksburg_crashes.parquet"
    traffic_file: str = "data/raw/blacksburg_traffic_volume.parquet"

    osm_graph_file: str = "data/processed/osm_drive_graph.graphml"
    osm_nodes_file: str = "data/processed/osm_nodes.parquet"
    osm_edges_file: str = "data/processed/osm_edges.parquet"
