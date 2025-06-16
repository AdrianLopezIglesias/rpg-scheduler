import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=528):
        super(PolicyNetwork, self).__init__()
        
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        self.move_head = nn.Linear(hidden_dim * 2, 1)
        self.treat_head = nn.Linear(hidden_dim * 2, 1)
        self.cure_head = nn.Linear(hidden_dim, 4)
        self.pass_head = nn.Linear(hidden_dim, 1)
        # New specialist for building centers
        self.build_head = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        node_embeddings = self.conv3(x, edge_index)
        graph_embedding = global_mean_pool(node_embeddings, batch)
        return node_embeddings, graph_embedding