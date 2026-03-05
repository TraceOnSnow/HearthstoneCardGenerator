import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool


class GNNModel(nn.Module):
    """Graph Neural Network model for Hearthstone card generation.

    Uses a multi-layer GCN/GAT architecture to encode card relationships
    and generate new card feature representations.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_gat: bool = False,
        gat_heads: int = 4,
    ) -> None:
        """Initialise the GNN model.

        Args:
            in_channels: Dimensionality of input node features.
            hidden_channels: Dimensionality of hidden layers.
            out_channels: Dimensionality of the output embedding.
            num_layers: Number of graph convolution layers.
            dropout: Dropout probability.
            use_gat: If True use GATConv layers, otherwise GCNConv.
            gat_heads: Number of attention heads (only used when use_gat=True).
        """
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.use_gat = use_gat

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            if use_gat:
                out_dim = hidden_channels // gat_heads if i < num_layers - 1 else hidden_channels
                self.convs.append(
                    GATConv(in_dim, out_dim, heads=gat_heads, concat=(i < num_layers - 1))
                )
            else:
                self.convs.append(GCNConv(in_dim, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the GNN.

        Args:
            x: Node feature matrix of shape (N, in_channels).
            edge_index: Graph connectivity in COO format of shape (2, E).
            batch: Batch vector of shape (N,) assigning each node to a graph.

        Returns:
            Graph-level or node-level embeddings of shape (*, out_channels).
        """
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        if batch is not None:
            x = global_mean_pool(x, batch)

        return self.fc(x)
