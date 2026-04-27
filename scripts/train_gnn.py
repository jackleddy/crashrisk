import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from crashrisk.config import Outputs
from crashrisk.gnn.dataset import build_edge_dataset
from crashrisk.gnn.model import GraphSAGE


def mae(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(a - b)).detach().cpu().item())


@torch.no_grad()
def eval_split(model, batch, idx, loss_fn):
    model.eval()

    log_mu = model(
        batch.data.x,
        batch.data.edge_index,
        batch.edge_u,
        batch.edge_v,
        batch.edge_attr,
        batch.log_exposure
    )

    loss = loss_fn(log_mu[idx], batch.y[idx]).detach().cpu().item()
    mu = torch.exp(log_mu[idx])
    yhat = mu

    return {
        "loss": float(loss),
        "mae_count": mae(yhat, batch.y[idx]),
    }


def main():
    out = Outputs()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    batch = build_edge_dataset(
        nodes_path=out.osm_nodes_file,
        edges_with_traffic_path=out.edges_with_traffic_file,
        train_edges_path=out.train_edges_file,
        device=device,
    )

    node_in = batch.data.x.size(1)
    edge_in = batch.edge_attr.size(1)

    # Put these params in src config file
    model = GraphSAGE(node_in=node_in, hidden=128, edge_in=edge_in, dropout=0.2).to(device)
    
    loss_fn = nn.PoissonNLLLoss(log_input=True, full=False, reduction="mean")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best_val = float("inf")
    best_path = out.gnn_model_file
    os.makedirs(os.path.dirname(best_path), exist_ok=True)

    for epoch in range(1, 31):
        model.train()
        opt.zero_grad()

        log_mu = model(
            batch.data.x, batch.data.edge_index,
            batch.edge_u, batch.edge_v,
            batch.edge_attr, batch.log_exposure
        )

        loss = loss_fn(log_mu[batch.train_idx], batch.y[batch.train_idx])
        loss.backward()
        opt.step()

        tr = eval_split(model, batch, batch.train_idx, loss_fn)
        va = eval_split(model, batch, batch.val_idx, loss_fn)
        te = eval_split(model, batch, batch.test_idx, loss_fn)

        if epoch % 100 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | "
                f"train loss {tr['loss']:.4f} mae {tr['mae_count']:.3f} | "
                f"val loss {va['loss']:.4f} mae {va['mae_count']:.3f} | "
                f"test loss {te['loss']:.4f} mae {te['mae_count']:.3f}"
            )

        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(model.state_dict(), best_path)


    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    with torch.no_grad():
        log_mu = model(
            batch.data.x, batch.data.edge_index,
            batch.edge_u, batch.edge_v,
            batch.edge_attr, batch.log_exposure
        )
        mu = torch.exp(log_mu).detach().cpu().numpy()
        y = batch.y.detach().cpu().numpy()
        exposure = np.exp(batch.log_exposure.detach().cpu().numpy())
        rate = mu / exposure

    pred = pd.DataFrame({
        "edge_id": batch.edge_ids,
        "y": y.astype(int),
        "mu_pred": mu,
        "exposure": exposure,
        "rate_pred": rate,
    })
    pred_path = out.gnn_edge_predictions_file
    pred.to_parquet(pred_path, index=False)
    # print("Wrote:", pred_path)

if __name__ == "__main__":
    main()
