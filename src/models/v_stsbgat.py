from torch import nn

from src.models.vision.semantic_context_encoder import SemanticContextEncoder
from src.models.gat.gat_layer import GATLayer
from src.models.transformer.temporal_encoder import TemporalEncoder


class VSTSBGT(nn.Module):
    def __init__(self, configs: dict):
        super(VSTSBGT, self).__init__()

        self.device = configs['device']

        te_configs = configs['obs_temporal_encoder']
        te_configs['device'] = self.device
        self.temporal_encoder = TemporalEncoder(d_in=te_configs['input_dim'],
                                                d_model=te_configs['dim_model'],
                                                nhead=te_configs['num_heads'],
                                                num_layers=te_configs['num_layers'],
                                                dropout=te_configs['dropout'])

        ze_configs = configs['z_temporal_encoder']
        ze_configs['device'] = self.device
        self.z_encoder = TemporalEncoder(d_in=ze_configs['input_dim'],
                                         d_model=ze_configs['dim_model'],
                                         nhead=ze_configs['num_heads'],
                                         num_layers=ze_configs['num_layers'],
                                         dropout=ze_configs['dropout'])

        gat_configs = configs['gat']
        self.gat_layer = GATLayer(gat_configs)

        map_encoder_configs = configs['map_encoder']
        self.semantic_context_encoder = SemanticContextEncoder(
                num_semantic_classes=map_encoder_configs["num_semantic_classes"],
                map_context_dim=map_encoder_configs.get("map_context_dim", 64,),
                cell_node_dim=map_encoder_configs.get("cell_node_dim", 64,),
                projection_dim=map_encoder_configs.get("projection_dim", 32,),
                pretrained=map_encoder_configs.get("pretrained", True,),
                stem_init=map_encoder_configs.get("stem_init", "random",),
                architecture=map_encoder_configs.get("architecture", "convnext_tiny",),
                freeze_backbone=map_encoder_configs.get("freeze_backbone", False,),
        )

        self.fc_out = nn.Linear(gat_configs['dim_model'], 1)

    def reset_parameters(self):
        """Reset parameters of the model."""
        nn.init.uniform_(self.fc_gat_out.weight, a=-1.0, b=1.0)
        # TODO: since the classes are not balanced, the weights can be initialized as pos/total

    def forward(self, x):
        seq_mask = x['seq_mask']  # Sequence mask for the historical observations
        vehicle_mask = x['vehicle_mask']
        cell_xy = x['hidden_ogm_cells']
        veh_feat = x['historical_adjacent_obs']
        semantic_map = x['map_obs']

        x_te = self.temporal_encoder(veh_feat, seq_mask, vehicle_mask)
        z_te = self.z_encoder(veh_feat, seq_mask, vehicle_mask)

        # -----------------------------------------
        # NEW:
        # semantic-map information goes upstream
        # of graph attention.
        # -----------------------------------------
        map_outputs = self.semantic_context_encoder(
            semantic_map=semantic_map,
            cell_xy=cell_xy,
        )
        cell_embedding = map_outputs["cell_embedding"]
        z_mask_embedding = map_outputs["z_mask_embedding"]

        gat_out, l2_loss, z_mask = self.gat_layer(x_te, z_te, cell_embedding, z_mask_embedding, x['edge_weights'], x['edge_index'])
        # gat_out: [B, N_cells, dim_model]. N_cells is 1 during training but can be >1 at
        # inference (e.g. predicting occupancy for every candidate cell of a frame at once).

        out_fc = self.fc_out(gat_out)

        return out_fc, None, l2_loss, z_mask
