import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv

torch.manual_seed(0)


class DropoutGCN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, edge_feature_dim, k, device, hidden1=16, hidden2=16, temperature=1.0):
        super(DropoutGCN, self).__init__()

        self.k = k
        self.temperature = temperature
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

        self.conv1 = GCNConv(input_dim, hidden1)
        self.conv2 = GCNConv(hidden1, hidden2)
        
        self.fc = nn.Linear(hidden2, output_dim)

    def forward(self, num_nodes, edge_index, edge_attr, x, node_mask, training=True):
        """
        Forward pass for NormalGCN.

        Args:
        - num_nodes (int): Number of nodes.
        - edge_index (Tensor): Edge indices.
        - edge_attr (Tensor): Edge attributes.
        - x (Tensor): Node features.
        - node_mask (Tensor): Node mask.
        - training (bool): Training mode.
        """
        # Apply GCN layers
        x = self.conv1(x, edge_index)  # First convolution
        x = F.relu(x)  # Apply ReLU activation
        x = self.conv2(x, edge_index)  # Second convolution
        x = F.relu(x)
        x = self.fc(x)
        x = x[node_mask != 0]
        
        return x