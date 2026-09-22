'''
GAT uses two layers of message passing between bi-partite graph nodes representing conventional attention calculation
and binary mask learning. Nodes also has two types of embeddings, one for conventional attention calculation and one for
binary mask learning. You can esstientially view this as two parallel GATs with same graph structure
but different node embeddings, learning to do different tasks.
'''

import torch
from torch import nn

from src.models.gat.gat import GATv2Conv


class GATLayer(nn.Module):
    def __init__(self, config):
        super(GATLayer, self).__init__()

        dim_adj_model = config['dim_obs_model']
        dim_cell_model = config['dim_cell_model']
        dim_model = config['dim_model']
        dim_edge = config['dim_edge']

        self.gat = GATv2Conv(in_channels=(dim_adj_model, dim_cell_model),
                             out_channels=dim_model,
                             heads=config['num_heads'],
                             dropout=config['dropout'],
                             concat=False,
                             edge_dim=dim_edge,
                             add_self_loops=False,
                             head_diversity_weight=config.get('head_diversity_weight', 0.0),
                             l0_loss_weights=config.get('l0_loss_weights', None))

    def forward(self, x_te_batch, x_te_z_batch, cell_batch, z_batch, edge_attr_batch, edge_index_batch):
        x_cell_batch = cell_batch
        x_cell_z_batch = z_batch

        gat_out_batch = []
        z_mask_batch = []
        ls_loss_batch = 0
        for (x_te, x_te_z, x_cell, x_cell_z, edge_index, edge_attr) in zip(x_te_batch, x_te_z_batch, x_cell_batch, x_cell_z_batch, edge_index_batch, edge_attr_batch):
            edge_index = edge_index.to(x_te.device)
            edge_attr = edge_attr.to(x_te.device)

            gat_out, l2_loss, z_mask = self.gat([x_te, x_te_z, x_cell, x_cell_z], edge_index, edge_attr)
            gat_out_batch.append(gat_out)
            z_mask_batch.append(z_mask)
            ls_loss_batch += l2_loss

        return torch.stack(gat_out_batch), ls_loss_batch, z_mask_batch
