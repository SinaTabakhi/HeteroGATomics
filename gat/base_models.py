from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, Linear

from utils import xavier_init, bias_init


class HeteroGNN(nn.Module):
    def __init__(self, hidden_channels: List[int], out_channels, num_layers, num_heads: int = 1, bias: bool = True,
                 dropout: float = 0.0):
        super().__init__()

        self.convs = nn.ModuleList()
        for layer_id in range(num_layers):
            conv = HeteroConv({
                ('patient', 'similar', 'patient'): GATConv(-1, hidden_channels[layer_id], heads=num_heads, concat=False,
                                                           dropout=dropout, add_self_loops=True, edge_dim=1),
                ('feature', 'similar', 'feature'): GATConv(-1, hidden_channels[layer_id], heads=num_heads, concat=False,
                                                           dropout=dropout, add_self_loops=True, edge_dim=2),
                ('feature', 'belong', 'patient'): GATConv((-1, -1), hidden_channels[layer_id], heads=num_heads,
                                                          concat=False, dropout=dropout, add_self_loops=False,
                                                          edge_dim=None),
            }, aggr='mean')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels[num_layers - 1], out_channels, bias=bias, weight_initializer='glorot')

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        return self.lin(x_dict['patient'])


class VCDN(nn.Module):
    def __init__(self, num_modalities: int, num_classes: int, hidden_dim: int) -> None:
        super().__init__()

        self.num_modalities = num_modalities
        self.num_classes = num_classes
        self.model = nn.Sequential(
            nn.Linear(pow(self.num_classes, self.num_modalities), hidden_dim),
            nn.LeakyReLU(0.25),
            nn.Linear(hidden_dim, self.num_classes),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.model.apply(xavier_init)
        self.model.apply(bias_init)

    def forward(self, multimodal_input: List[torch.Tensor]) -> torch.Tensor:
        for modality in range(self.num_modalities):
            multimodal_input[modality] = torch.sigmoid(multimodal_input[modality])
        x = torch.reshape(
            torch.matmul(multimodal_input[0].unsqueeze(-1), multimodal_input[1].unsqueeze(1)),
            (-1, pow(self.num_classes, 2), 1),
        )
        for modality in range(2, self.num_modalities):
            x = torch.reshape(
                torch.matmul(x, multimodal_input[modality].unsqueeze(1)), (-1, pow(self.num_classes, modality + 1), 1)
            )
        input_tensor = torch.reshape(x, (-1, pow(self.num_classes, self.num_modalities)))
        output = self.model(input_tensor)

        return output
