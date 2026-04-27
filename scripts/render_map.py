from crashrisk.config import Outputs
from crashrisk.visualization.risk_map import render_risk_map

def main():
    out = Outputs()

    render_risk_map(
        osm_edges_path=out.edges_with_traffic_file,   # has edge_id + geometry
        crashes_path=out.crashes_snapped_file,        # raw crash points (snapped file is fine; still has geometry)
        predictions_path=out.gnn_edge_predictions_file,
        out_html_path=out.risk_map_html_file,
        pred_value="mu_pred",                         # compare against crash hotspot counts
        show_raw_roads=True,
        show_risk_roads=True,
        show_crashes=True,
        crash_marker_cluster=False,
        risk_quantile_clip=0.99,
    )
    print(f"Wrote: {out.risk_map_html_file}")


if __name__ == "__main__":
    main()