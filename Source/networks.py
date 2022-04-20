#----------------------------------------------------------------------
# Definition of the neural network architectures
# Author: Pablo Villanueva Domingo
# Last update: 10/11/21
#----------------------------------------------------------------------

import torch
from torch.nn import Sequential, Linear, ReLU, ModuleList
from torch_geometric.nn import MessagePassing, GCNConv, PPFConv, MetaLayer, EdgeConv
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_cluster import knn_graph, radius_graph
from torch_scatter import scatter_mean, scatter_sum, scatter_add, scatter_max, scatter_min
import numpy as np
from Source.constants import *


#------------------------------
# Architectures considered:
#   DeepSet
#   Metalayer (graph network)
#
#-----------------------------

#--------------------------------------------
# Message passing architecture
# See pytorch-geometric documentation for more info
# pytorch-geometric.readthedocs.io/
#--------------------------------------------


# Node model for the MetaLayer
class NodeModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels, link_r):
        super(NodeModel, self).__init__()
        #self.node_mlp_1 = Sequential(Linear(in_channels,hidden_channels),  LeakyReLU(0.2), Linear(hidden_channels,hidden_channels),LeakyReLU(0.2), Linear(mid_channels,out_channels))
        #self.node_mlp_2 = Sequential(Linear(303,500), LeakyReLU(0.2), Linear(500,500),LeakyReLU(0.2), Linear(500,1))

        if only_positions:
            inchan = in_channels + 3
            secchan = 3*latent_channels+in_channels
        else:
            inchan = (in_channels-3)*2+3
            secchan = 3*latent_channels+2+in_channels-3


        self.mlp1 = Sequential(Linear(inchan, hidden_channels),
                              ReLU(),
                              Linear(hidden_channels, hidden_channels),
                              ReLU(),
                              Linear(hidden_channels, latent_channels))

        self.mlp2 = Sequential(Linear(secchan, hidden_channels),
                              ReLU(),
                              Linear(hidden_channels, hidden_channels),
                              ReLU(),
                              Linear(hidden_channels, latent_channels))

        self.link_r = link_r

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        #x_node, x_neigh = x[row], x[col]

        #edge_attr = torch.sqrt(torch.sum((x[row,:3]-x[col,:3])**2.,axis=1))
        #edge_attr = edge_attr.view(-1,1)
        edge_attr = x[row,:3]-x[col,:3]

        # correct distance for boundaries
        for coord in range(3):
            edge_attr[edge_attr[:,coord]>self.link_r,coord]-=1.
            edge_attr[-edge_attr[:,coord]>self.link_r,coord]+=1.

        # define interaction tensor; every pair contains features from input and
        # output node together with
        if only_positions:
            out = torch.cat([x[row], edge_attr], dim=1)
        else:
            out = torch.cat([x[row,3:], x[col,3:], edge_attr], dim=1)

        # take interaction feature tensor and embedd it into another tensor
        #out = self.node_mlp_1(out)
        out = self.mlp1(out)

        # compute the mean,sum and max of each embed feature tensor for each node
        out1 = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out2 = scatter_max(out, col, dim=0, dim_size=x.size(0))[0]
        out3 = scatter_add(out, col, dim=0, dim_size=x.size(0))
        #out3 = scatter_min(out, col, dim=0, dim_size=x.size(0))[0]

        # every node contains a feature tensor with the pooling of the messages from
        # neighbors, its own state, and a global feature
        if only_positions:
            out = torch.cat([x, out1, out2, out3], dim=1)
        else:
            out = torch.cat([x[:,3:], out1, out2, out3, u[batch]], dim=1)
        #print("node post", out.shape)

        out = self.mlp2(out)

        # In this way, it is traslational equivariant
        out = torch.cat([x[:,:3], out], dim=1)

        return out



#--------------------------------------------
# General Graph Neural Network architecture
#--------------------------------------------
class ModelGNN(torch.nn.Module):
    def __init__(self, use_model, node_features, n_layers, link_r, hidden_channels=300, latent_channels=100, loop=True):
        super(ModelGNN, self).__init__()

        # Graph layers
        layers = []
        in_channels = node_features
        for i in range(n_layers):

            # Choose the model
            if use_model=="DeepSet":
                lay = Sequential(
                    Linear(in_channels, hidden_channels),
                	ReLU(),
                	Linear(hidden_channels, hidden_channels),
                	ReLU(),
                	Linear(hidden_channels, latent_channels))


            elif use_model=="MetaNet":
                lay = MetaLayer(node_model=NodeModel(in_channels, hidden_channels, latent_channels, link_r))

            else:
                print("Model not known...")

            layers.append(lay)
            in_channels = latent_channels+3


        self.layers = ModuleList(layers)


        if only_positions:
            lin_in = (in_channels-3)*3
        else:
            lin_in = (in_channels-3)*3+2

        self.lin = Sequential(Linear(lin_in, 2*latent_channels),
                              ReLU(),
                              Linear(2*latent_channels, latent_channels),
                              ReLU(),
                              Linear(latent_channels, latent_channels),
                              ReLU(),
                              Linear(latent_channels, 4))

        self.link_r = link_r
        self.pooled = 0.
        self.loop = loop
        self.namemodel = use_model

    def forward(self, data):

        x, pos, batch, u = data.x, data.x[:,:3], data.batch, data.u

        # Get edges using positions by computing the kNNs or the neighbors within a radius
        #edge_index = knn_graph(pos, k=self.link_r, batch=batch, loop=self.loop)
        #edge_index = radius_graph(pos, r=self.link_r, batch=batch, loop=self.loop)
        edge_index = data.edge_index

        #if self.namemodel=="MetaNet":
        #    edge_attr = get_edge_attr(edge_index,pos)

        # Start message passing
        for layer in self.layers:
            if self.namemodel=="DeepSet":
                x = layer(x)
            elif self.namemodel=="MetaNet":
                x, dumb, u = layer(x, edge_index, None, u, batch)
            x = x.relu()


        # Mix different global pooling layers
        addpool = global_add_pool(x[:,3:], batch) # [num_examples, hidden_channels]
        meanpool = global_mean_pool(x[:,3:], batch)
        maxpool = global_max_pool(x[:,3:], batch)

        if only_positions:
            self.pooled = torch.cat([addpool, meanpool, maxpool], dim=1)
        else:
            self.pooled = torch.cat([addpool, meanpool, maxpool, u], dim=1)

        # Final linear layer
        out = self.lin(self.pooled)

        return out

def get_edge_attr(edge_index,pos):

    edge_attr = torch.zeros((edge_index.shape[1],1), dtype=torch.float32, device=device)
    for i, nodein in enumerate(edge_index[0,:]):
        nodeout = edge_index[1,i]
        edge_attr[i,0]=torch.sqrt(torch.sum((pos[nodein]-pos[nodeout])**2.,axis=0))
    return edge_attr
