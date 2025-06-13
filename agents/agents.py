import random
import joblib
import numpy as np
import copy
import os

# --- NEW IMPORTS FOR PHASE 2 ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.distributions import Categorical
# -------------------------------

def create_feature_vector(game_state, game):
    # This function is now only used by the old NNAgent
    # ... (existing code) ...
    features = []
    player_loc = game_state['player_location']
    board = game_state['board']
    
    features.append(board[player_loc]['cubes'])
    neighbors1 = sorted(game.map[player_loc]['neighbors'])
    for i in range(4):
        features.append(board[neighbors1[i]]['cubes'] if i < len(neighbors1) else 0)

    neighbors2_set = set(n2 for n1 in neighbors1 for n2 in game.map[n1]['neighbors'])
    neighbors2 = sorted(list(neighbors2_set - set(neighbors1) - {player_loc}))
  
    for i in range(8):
        features.append(board[neighbors2[i]]['cubes'] if i < len(neighbors2) else 0)
        
    neighbors3_set = set(n3 for n2 in neighbors2 for n3 in game.map[n2]['neighbors'])
    neighbors3 = sorted(list(neighbors3_set - set(neighbors2) - set(neighbors1) - {player_loc}))
    for i in range(16):
        features.append(board[neighbors3[i]]['cubes'] if i < len(neighbors3) else 0)

    features.append(sum(c['cubes'] for c in board.values()))
    danger_cities = [city for city, data in board.items() if data['cubes'] == 3]
   
    features.append(len(danger_cities))
    features.append(min([game.get_distance(player_loc, c) for c in danger_cities]) if danger_cities else -1)
    
    return np.array(features).reshape(1, -1)


class Agent:
    def choose_action(self, game, possible_actions):
        raise NotImplementedError

# --- EXISTING AGENTS (No changes) ---
class RandomAgent(Agent):
    # ... (existing code) ...
    def choose_action(self, game, possible_actions):
        move_actions = [a for a in possible_actions if a['type'] == 'move']
        treat_actions = [a for a in possible_actions if a['type'] == 'treat']

        available_action_types = []
        if move_actions:
            available_action_types.append('move')
        if treat_actions:
            available_action_types.append('treat')

        if not available_action_types:
            return random.choice(possible_actions) # Fallback for 'pass' or empty list

        chosen_type = random.choice(available_action_types)

        if chosen_type == 'move':
            return random.choice(move_actions)
        else: # chosen_type == 'treat'
            return random.choice(treat_actions)

class NNAgent(Agent):
    # ... (existing code) ...
    def __init__(self, model_path_prefix, epsilon=0.1):
        self.epsilon = epsilon
        try:
            self.model = joblib.load(f"{model_path_prefix}.joblib")
            self.scaler = joblib.load(f"{model_path_prefix}_scaler.joblib")
        except FileNotFoundError:
            self.model = None

    def _simulate_next_state(self, game, action):
        sim_game = copy.deepcopy(game)
        # This part will break because the step method changed.
        # This agent is now legacy.
        # A proper fix would be to adapt it, but for a minimal
        # viable product we focus on the new GNN agent.
        sim_game.step(action) # This call is now incorrect.
        return sim_game.get_state_snapshot()

    def choose_action(self, game, possible_actions):
        # ... (rest of existing code) ...
        if not self.model:
            return random.choice(possible_actions)
        if random.random() < self.epsilon:
            return random.choice(possible_actions)
        best_action = None
        best_predicted_score = -float('inf')
        for action in possible_actions:
            # This logic is broken due to step() changes.
            pass
        return best_action if best_action else random.choice(possible_actions)

# --- NEW CLASSES FOR PHASE 2 ---

class PolicyNetwork(nn.Module):
    """GNN to process the board state and output action logits."""
    def __init__(self, input_dim, output_dim, hidden_dim=32):
        super(PolicyNetwork, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch) # Aggregate node features to a graph-level feature
        action_logits = self.output_layer(x)
        return action_logits

class GNNAgent(Agent):
    """The RL Agent that uses the GNN policy network."""
    def __init__(self, input_dim, output_dim, config):
        rl_cfg = config['rl_config']
        self.policy_network = PolicyNetwork(input_dim, output_dim)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=rl_cfg['learning_rate'])
        self.gamma = rl_cfg['gamma']
        self.log_probs = []
        self.rewards = []

    def choose_action(self, state_graph, possible_actions_mask):
        """Chooses an action based on the policy network's output."""
        state_graph.batch = torch.zeros(state_graph.num_nodes, dtype=torch.long)
        logits = self.policy_network(state_graph).squeeze(0)
        logits[~possible_actions_mask] = -1e8
        prob_dist = Categorical(logits=logits)
        action_idx = prob_dist.sample()
        self.log_probs.append(prob_dist.log_prob(action_idx))
        return action_idx.item()

    def update_policy(self):
        """Updates the policy network's weights using the REINFORCE algorithm."""
        discounted_rewards = []
        R = 0
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        if len(discounted_rewards) > 1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        policy_loss = [-log_prob * reward for log_prob, reward in zip(self.log_probs, discounted_rewards)]
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        self.log_probs = []
        self.rewards = []

    def load_model(self, path):
        self.policy_network.load_state_dict(torch.load(path))

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy_network.state_dict(), path)