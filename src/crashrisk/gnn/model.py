import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, node_in: int, hidden: int, edge_in: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = SAGEConv(node_in, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.dropout = dropout

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden + edge_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def encode_nodes(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index)
        h = F.relu(h)
        return h
    
    def forward(self, x, edge_index, edge_u, edge_v, edge_attr, log_exposure):
        h = self.encode_nodes(x, edge_index)
        hu = h[edge_u]
        hv = h[edge_v]
        z = torch.cat([hu, hv, edge_attr], dim=1)
        s = self.edge_mlp(z).squeeze(-1)  # [M]
        log_mu = log_exposure + s
        return log_mu