import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import BatchNorm

activation_function_dict = {"relu": F.relu,
                            "leaky_relu": F.leaky_relu}


class GATv2Net(torch.nn.Module):
    """AEV-PLIG GATv2 net. Depth (number_GNN_layers) and the MLP head (mlp_dims)
    are configurable for parameter scaling; defaults reproduce the paper's 7.55M
    model (5 GNN layers, hidden 256 x 3 heads, MLP 1024->512->256)."""

    def __init__(self, node_feature_dim, edge_feature_dim, config):
        super(GATv2Net, self).__init__()

        # scaling knobs (back-compat defaults = paper config)
        self.number_GNN_layers = getattr(config, "number_GNN_layers", 5)
        mlp_dims = getattr(config, "mlp_dims", [1024, 512, 256])
        self.act = config.activation_function
        self.activation = activation_function_dict[self.act]

        self.GNN_layers = nn.ModuleList()
        self.BN_layers = nn.ModuleList()

        input_dim = node_feature_dim
        head = config.head
        hidden_dim = config.hidden_dim

        self.GNN_layers.append(GATv2Conv(input_dim, hidden_dim, heads=head, edge_dim=edge_feature_dim))
        self.BN_layers.append(BatchNorm(hidden_dim * head))
        for i in range(1, self.number_GNN_layers):
            self.GNN_layers.append(GATv2Conv(hidden_dim * head, hidden_dim, heads=head, edge_dim=edge_feature_dim))
            self.BN_layers.append(BatchNorm(hidden_dim * head))

        final_dim = hidden_dim * head

        # MLP head (variable depth): [max||mean pool -> 2*final_dim] -> mlp_dims -> 1
        self.fc_layers = nn.ModuleList()
        self.fc_bn = nn.ModuleList()
        prev = final_dim * 2
        for d in mlp_dims:
            self.fc_layers.append(nn.Linear(prev, d))
            self.fc_bn.append(nn.BatchNorm1d(d))
            prev = d
        self.out = nn.Linear(prev, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        for layer, bn in zip(self.GNN_layers, self.BN_layers):
            x = layer(x, edge_index, edge_attr)
            x = self.activation(x)
            x = bn(x)

        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        for fc, bn in zip(self.fc_layers, self.fc_bn):
            x = fc(x)
            x = self.activation(x)
            x = bn(x)
        return self.out(x)


    
