import geopandas as gpd
import osmnx as ox
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union


def get_region(place_name: str, buffer_m: float = 0.0, simplify_m: float = 10.0):
    ox.settings.use_cache = True
    ox.settings.log_console = False

    gdf = ox.geocode_to_gdf(place_name)
    geom = unary_union(gdf.geometry.values)

    if not isinstance(geom, (MultiPolygon, Polygon)):
        raise ValueError(f"Unsupported geometry for {place_name}: {type(geom)}")

    region = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
    region_utm = region.to_crs(region.estimate_utm_crs())
    if buffer_m and buffer_m > 0:
        region_utm["geometry"] = region_utm.buffer(buffer_m)

    if simplify_m and simplify_m > 0:
        region_utm["geometry"] = region_utm.simplify(simplify_m)

    region_wgs84 = region_utm.to_crs("EPSG:4326")
    return region_wgs84.geometry.iloc[0]
