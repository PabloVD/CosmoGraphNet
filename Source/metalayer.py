import torch
import torch.nn.functional as F
from torch_cluster import knn_graph, radius_graph
from torch.nn import Sequential, Linear, ReLU, ModuleList
from torch_geometric.nn import MessagePassing, MetaLayer, LayerNorm
from torch_scatter import scatter_mean, scatter_sum, scatter_max, scatter_min, scatter_add
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
#import torch.utils.checkpoint
from Source.constants import *

#use_checkpoints = False

class EdgeModel(torch.nn.Module):
    def __init__(self, node_in, node_out, edge_in, edge_out, hid_channels, residuals=True, norm=False):
        super().__init__()

        self.residuals = residuals
        self.norm = norm

        layers = [Linear(node_in*2 + edge_in, hid_channels),
                  ReLU(),
                  Linear(hid_channels, edge_out)]
        if self.norm:  layers.append(LayerNorm(edge_out))

        self.edge_mlp = Sequential(*layers)


    def forward(self, src, dest, edge_attr, u, batch):
        # src, dest: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.

        out = torch.cat([src, dest, edge_attr], dim=1)
        #out = torch.cat([src, dest, edge_attr, u[batch]], 1)
        out = self.edge_mlp(out)
        #out = torch.utils.checkpoint.checkpoint_sequential(self.edge_mlp, 2, out)
        if self.residuals:
            out = out + edge_attr
        return out

class NodeModel(torch.nn.Module):
    def __init__(self, node_in, node_out, edge_in, edge_out, hid_channels, residuals=True, norm=False):
        super().__init__()

        self.residuals = residuals
        self.norm = norm

        layers = [Linear(node_in + 3*edge_out + 1, hid_channels),
                  ReLU(),
                  Linear(hid_channels, node_out)]
        if self.norm:  layers.append(LayerNorm(node_out))

        self.node_mlp = Sequential(*layers)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        #out = torch.cat([x[row], edge_attr], dim=1)
        #out = self.node_mlp_1(out)
        out = edge_attr
        out1 = scatter_add(out, col, dim=0, dim_size=x.size(0))
        out2 = scatter_max(out, col, dim=0, dim_size=x.size(0))[0]
        out3 = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out1, out2, out3, u[batch]], dim=1)
        #out = torch.cat([x, out], dim=1)
        #out = self.node_mlp(out)
        #if use_checkpoints:
        #    out = torch.utils.checkpoint.checkpoint_sequential(self.node_mlp, 2, out)
        #else:
        out = self.node_mlp(out)
        if self.residuals:
            out = out + x
        return out

class EdgeModelIn(torch.nn.Module):
    def __init__(self, node_in, node_out, edge_in, edge_out, hid_channels, norm=False):
        super().__init__()

        self.norm = norm

        layers = [Linear(edge_in, hid_channels),
                  ReLU(),
                  Linear(hid_channels, edge_out)]
        if self.norm:  layers.append(LayerNorm(edge_out))

        self.edge_mlp = Sequential(*layers)


    def forward(self, src, dest, edge_attr, u, batch):

        out = self.edge_mlp(edge_attr)

        return out

class NodeModelIn(torch.nn.Module):
    def __init__(self, node_in, node_out, edge_in, edge_out, hid_channels, norm=False):
        super().__init__()

        self.norm = norm

        layers = [Linear(3*edge_out + 1, hid_channels),
                  ReLU(),
                  Linear(hid_channels, node_out)]
        if self.norm:  layers.append(LayerNorm(node_out))

        self.node_mlp = Sequential(*layers)

    def forward(self, x, edge_index, edge_attr, u, batch):

        row, col = edge_index

        out = edge_attr
        #out = scatter_add(out, col, dim=0, dim_size=x.size(0))
        out1 = scatter_add(out, col, dim=0, dim_size=x.size(0))
        out2 = scatter_max(out, col, dim=0, dim_size=x.size(0))[0]
        out3 = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([out1, out2, out3, u[batch]], dim=1)
        #out = torch.cat([out, u[batch]], dim=1)

        out = self.node_mlp(out)

        return out

# Graph Neural Network architecture, based on the Interaction Network (arXiv:1612.00222, arXiv:2002.09405)
class GNN(torch.nn.Module):
    def __init__(self, node_features, n_layers, hidden_channels, linkradius, dim_out, only_positions, residuals=True):
        super().__init__()

        self.n_layers = n_layers
        #self.loop = False
        self.link_r = linkradius
        self.dim_out = dim_out
        self.only_positions = only_positions

        # Input node features (0 if only_positions is used)
        node_in = node_features
        # Input edge features: |p_i-p_j|, p_i*p_j, p_i*(p_i-p_j)
        edge_in = 3
        node_out = hidden_channels
        edge_out = hidden_channels
        hid_channels = hidden_channels

        layers = []

        # Encoder layer
        # If use only positions, node features are created from the aggregation of edge attritbutes of neighbors
        if self.only_positions:
            inlayer = MetaLayer(node_model=NodeModelIn(node_in, node_out, edge_in, edge_out, hid_channels),
                                edge_model=EdgeModelIn(node_in, node_out, edge_in, edge_out, hid_channels))

        else:
            inlayer = MetaLayer(node_model=NodeModel(node_in, node_out, edge_in, edge_out, hid_channels, residuals=False),
                                edge_model=EdgeModel(node_in, node_out, edge_in, edge_out, hid_channels, residuals=False))

        layers.append(inlayer)

        # Change input node and edge feature sizes
        node_in = node_out
        edge_in = edge_out

        # Hidden graph layers
        for i in range(n_layers-1):

            lay = MetaLayer(node_model=NodeModel(node_in, node_out, edge_in, edge_out, hid_channels, residuals=residuals),
                            edge_model=EdgeModel(node_in, node_out, edge_in, edge_out, hid_channels, residuals=residuals))
            layers.append(lay)

        self.layers = ModuleList(layers)

        self.outlayer = Sequential(Linear(3*node_out+1, hid_channels),
                              ReLU(),
                              Linear(hid_channels, hid_channels),
                              ReLU(),
                              Linear(hid_channels, hid_channels),
                              ReLU(),
                              Linear(hid_channels, self.dim_out))


    def get_graph(self, data):

        """pos = data.x[:,:3]

        # Get edges
        edge_index = radius_graph(pos, r=self.link_r, batch=data.batch, loop=self.loop)
        #edge_index = data.edge_index

        row, col = edge_index

        # Edge features
        diff = (pos[row]-pos[col])/self.link_r
        edge_attr = torch.cat([diff, torch.norm(diff, dim=1, keepdim=True)], dim=1)"""

        edge_index, edge_attr = data.edge_index, data.edge_attr

        # Node features
        if self.only_positions:
            h = torch.zeros_like(data.x[:,0])
        else:
            h = data.x

        return h, edge_index, edge_attr


    def forward(self, data):
    #def forward(self, x, batch):

        #h, edge_index, edge_attr = self.get_graph(data)
        h, edge_index, edge_attr, u = data.x, data.edge_index, data.edge_attr, data.u

        for layer in self.layers:
            h, edge_attr, _ = layer(h, edge_index, edge_attr, u, data.batch)

        addpool = global_add_pool(h, data.batch)
        meanpool = global_mean_pool(h, data.batch)
        maxpool = global_max_pool(h, data.batch)

        out = torch.cat([addpool,meanpool,maxpool,u], dim=1)

        # Final linear layer
        out = self.outlayer(out)

        return out
