import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, critic_hidden_dim=128, actor_hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        
        # GCN Layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # --- Actor Heads (Multi-layer) ---
        self.move_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, actor_hidden_dim),
            nn.ReLU(),
            nn.Linear(actor_hidden_dim, 1)
        )
        self.treat_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, actor_hidden_dim),
            nn.ReLU(),
            nn.Linear(actor_hidden_dim, 1)
        )
        self.cure_head = nn.Sequential(
            nn.Linear(hidden_dim, actor_hidden_dim),
            nn.ReLU(),
            nn.Linear(actor_hidden_dim, 4)
        )
        self.pass_head = nn.Sequential(
            nn.Linear(hidden_dim, actor_hidden_dim),
            nn.ReLU(),
            nn.Linear(actor_hidden_dim, 1)
        )
        self.build_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, actor_hidden_dim),
            nn.ReLU(),
            nn.Linear(actor_hidden_dim, 1)
        )

        # --- Critic Head (Multi-layer) ---
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, critic_hidden_dim),
            nn.ReLU(),
            nn.Linear(critic_hidden_dim, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = self.conv2(x, edge_index)
        # x = F.relu(x)
        node_embeddings = self.conv3(x, edge_index)
        graph_embedding = global_mean_pool(node_embeddings, batch)

        state_value = self.value_head(graph_embedding)
        
        return node_embeddings, graph_embedding, state_value.squeeze(-1)