# This file was mostly written by chat bc I couldn't be bothered. We may have to rewrite it eventually
# but it works so ... fine for now?
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import geopandas as gpd
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from shapely.geometry import mapping
from tqdm import tqdm
from urllib3.util.retry import Retry

OUTER_RETRY_ATTEMPTS = 3
OUTER_RETRY_BACKOFF_S = 1.0


@dataclass(frozen=True)
class ArcGISLayer:
    layer_url: str

    @property
    def query_url(self) -> str:
        return self.layer_url.rstrip("/") + "/query"

    def _build_session(self) -> requests.Session:
        retry = Retry(
            total=5,
            connect=5,
            read=5,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=None,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session = requests.Session()
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update({"User-Agent": "crashrisk/0.1"})
        return session

    def _post_json(
        self,
        data: Dict[str, Any],
        timeout_s: int = 120,
        operation: str = "request",
    ) -> Dict[str, Any]:
        last_exc: requests.RequestException | None = None

        for attempt in range(1, OUTER_RETRY_ATTEMPTS + 1):
            try:
                with self._build_session() as session:
                    response = session.post(self.query_url, data=data, timeout=timeout_s)
                    response.raise_for_status()
                    return response.json()
            except requests.RequestException as exc:
                last_exc = exc
                if attempt == OUTER_RETRY_ATTEMPTS:
                    break
                time.sleep(OUTER_RETRY_BACKOFF_S * attempt)

        raise RuntimeError(
            f"ArcGIS {operation} failed for {self.query_url} after "
            f"{OUTER_RETRY_ATTEMPTS} attempts: {last_exc}"
        ) from last_exc

    def _post(self, data: Dict[str, Any], timeout_s: int = 120) -> Dict[str, Any]:
        return self._post_json(data=data, timeout_s=timeout_s, operation="_post")

    def count(self, where: str, geometry_esrijson: str, geometry_type: str) -> int:
        payload = {
            "f": "json",
            "where": where,
            "returnCountOnly": "true",
            "geometry": geometry_esrijson,
            "geometryType": geometry_type,
            "spatialRel": "esriSpatialRelIntersects",
            "inSR": "4326",
        }
        js = self._post(payload)
        if "count" not in js:
            raise RuntimeError(f"ArcGIS count() failed: {js}")
        return int(js["count"])

    def query_geojson_paged(
        self,
        where: str,
        out_fields: Sequence[str],
        geometry_esrijson: str,
        geometry_type: str = "esriGeometryPolygon",
        out_sr: int = 4326,
        page_size: int = 2000,
        geometry_precision: int = 6,
        show_progress: bool = True,
    ) -> gpd.GeoDataFrame:
        fields = ",".join(out_fields)
        result_offset = 0
        total = self.count(
            where=where,
            geometry_esrijson=geometry_esrijson,
            geometry_type=geometry_type,
        )
        frames: List[gpd.GeoDataFrame] = []

        it = range(0, total, page_size)
        if show_progress:
            it = tqdm(it, desc="Downloading", unit="page")

        for _ in it:
            payload = {
                "f": "geojson",
                "where": where,
                "outFields": fields,
                "returnGeometry": "true",
                "geometry": geometry_esrijson,
                "geometryType": geometry_type,
                "spatialRel": "esriSpatialRelIntersects",
                "inSR": str(out_sr),
                "outSR": str(out_sr),
                "resultOffset": str(result_offset),
                "resultRecordCount": str(page_size),
                "geometryPrecision": str(geometry_precision),
                "returnZ": "false",
                "returnM": "false",
            }
            gj = self._post_json(
                data=payload,
                timeout_s=180,
                operation=f"page fetch at offset {result_offset}",
            )

            feats = gj.get("features", [])
            if not feats:
                break

            gdf = gpd.GeoDataFrame.from_features(feats, crs=f"EPSG:{out_sr}")
            frames.append(gdf)
            result_offset += page_size

        if not frames:
            return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{out_sr}")

        out = pd.concat(frames, ignore_index=True)
        return gpd.GeoDataFrame(out, geometry="geometry", crs=f"EPSG:{out_sr}")


def shapely_polygon_to_esri_polygon_json(poly) -> str:
    geo = mapping(poly)
    rings: List[List[List[float]]] = []

    def add_polygon_coords(coords: Iterable[Tuple[float, float]]) -> None:
        ring = [[float(x), float(y)] for (x, y) in coords]
        rings.append(ring)

    if geo["type"] == "Polygon":
        for ring_coords in geo["coordinates"]:
            add_polygon_coords(ring_coords)
    elif geo["type"] == "MultiPolygon":
        for poly_coords in geo["coordinates"]:
            for ring_coords in poly_coords:
                add_polygon_coords(ring_coords)
    else:
        raise ValueError(f"Unsupported geometry type: {geo['type']}")

    esri = {"rings": rings, "spatialReference": {"wkid": 4326}}
    return json.dumps(esri)
