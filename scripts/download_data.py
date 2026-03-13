from pathlib import Path

from crashrisk.config import (
    Outputs,
    CRASH_OUTFIELDS,
    CRASH_PAGE_SIZE,
    CRASH_YEAR_MAX,
    CRASH_YEAR_MIN,
    CRASH_LAYER_URL,
    REGION_BUFFER,
    REGION_NAME,
    TRAFFIC_OUTFIELDS,
    TRAFFIC_PAGE_SIZE,
    TRAFFIC_VOLUME_LAYER_URL,
)
from crashrisk.ingest.arcgis import ArcGISLayer, shapely_polygon_to_esri_polygon_json
from crashrisk.ingest.regions import get_region

def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def main() -> None:
    out = Outputs()

    print(f"Region: {REGION_NAME}")
    region_poly = get_region(REGION_NAME, buffer_m=REGION_BUFFER) 
    geom = shapely_polygon_to_esri_polygon_json(region_poly)

    # Crash filters. More will probably be added later, but idk if year even matters.
    where_crash = f"CRASH_YEAR >= '{CRASH_YEAR_MIN}' AND CRASH_YEAR <= '{CRASH_YEAR_MAX}'"

    # Getting crash data
    crash_layer = ArcGISLayer(CRASH_LAYER_URL)
    print("Downloading crash points ...")
    crashes = crash_layer.query_geojson_paged(
        where=where_crash,
        out_fields=CRASH_OUTFIELDS,
        geometry_esrijson=geom,
        geometry_type="esriGeometryPolygon",
        page_size=CRASH_PAGE_SIZE,
    )
    print(f"Crash records pulled: {len(crashes):,}")
    
    ensure_parent(out.crashes_file)
    crashes.to_parquet(out.crashes_file, index=False)
    print(f"Wrote to: {out.crashes_file}")


    # Traffic filters
    where_traffic = "1=1" # placeholder

    # Getting traffic data
    traffic_layer = ArcGISLayer(TRAFFIC_VOLUME_LAYER_URL)
    print("Downloading traffic volume ...")
    traffic = traffic_layer.query_geojson_paged(
        where=where_traffic,
        out_fields=TRAFFIC_OUTFIELDS,
        geometry_esrijson=geom,
        geometry_type="esriGeometryPolygon",
        page_size=TRAFFIC_PAGE_SIZE,
    )
    print(f"Traffic records pulled: {len(traffic):,}")

    ensure_parent(out.traffic_file)
    traffic.to_parquet(out.traffic_file, index=False)
    print(f"Wrote: {out.traffic_file}")


if __name__ == "__main__":
    main()
