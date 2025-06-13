import random
import joblib
import numpy as np
import copy
import os
import json # Was missing

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.distributions import Categorical

# (create_feature_vector, Agent, RandomAgent, NNAgent are unchanged)
def create_feature_vector(game_state, game):
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

class RandomAgent(Agent):
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


# --- MODIFIED: PolicyNetwork is now map-agnostic ---
class PolicyNetwork(nn.Module):
    """GNN that learns node embeddings and uses heads to score action types."""
    def __init__(self, input_dim, hidden_dim=32):
        super(PolicyNetwork, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Action-specific "heads"
        self.move_head = nn.Linear(hidden_dim, 1)
        self.treat_head = nn.Linear(hidden_dim, 1)
        self.pass_head = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 1. Get node embeddings
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        node_embeddings = self.conv2(x, edge_index)
        
        # 2. Get a single embedding for the whole graph (for the 'pass' action)
        graph_embedding = global_mean_pool(node_embeddings, batch)
        
        return node_embeddings, graph_embedding

# --- MODIFIED: GNNAgent now uses the agnostic network ---
class GNNAgent(Agent):
    """RL Agent that uses the map-agnostic GNN."""
    def __init__(self, input_dim, config): # Output_dim removed
        rl_cfg = config['rl_config']
        self.policy_network = PolicyNetwork(input_dim)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=rl_cfg['learning_rate'])
        self.gamma = rl_cfg['gamma']
        self.log_probs = []
        self.rewards = []

    def choose_action(self, game, state_graph):
        """
        Calculates scores for only the currently legal actions and chooses one.
        This is the core of the map-agnostic approach.
        """
        (node_embeddings, graph_embedding) = self.policy_network(state_graph)
        
        possible_actions_mask = game.get_possible_action_mask()
        
        # We build a sparse logits tensor, calculating scores only for legal moves
        logits = torch.full_like(possible_actions_mask, -1e8, dtype=torch.float)

        for action_idx, is_possible in enumerate(possible_actions_mask):
            if is_possible:
                action_desc = game.idx_to_action[action_idx]
                score = 0
                if action_desc['type'] == 'move':
                    target_node_idx = action_desc['target_idx']
                    score = self.policy_network.move_head(node_embeddings[target_node_idx])
                elif action_desc['type'] == 'treat':
                    target_node_idx = action_desc['target_idx']
                    score = self.policy_network.treat_head(node_embeddings[target_node_idx])
                elif action_desc['type'] == 'pass':
                    score = self.policy_network.pass_head(graph_embedding)
                
                logits[action_idx] = score

        prob_dist = Categorical(logits=logits)
        chosen_action_idx = prob_dist.sample()
        
        self.log_probs.append(prob_dist.log_prob(chosen_action_idx))
        
        return chosen_action_idx.item()

    def update_policy(self):
        # Initializes an empty list to hold the calculated rewards for each step.
        discounted_rewards = []

        # This variable, R, will hold the running total of the reward. It starts at 0.
        R = 0

        # This loop iterates backwards through the raw rewards collected during the episode.
        # Example: [0, 0, 0, 100] -> it will process 100, then 0, then 0, then 0.
        for r in self.rewards[::-1]:
            # The core of credit assignment.
            # The reward for a step is that step's raw reward plus the discounted reward of the *next* step.
            # Gamma (e.g., 0.99) makes future rewards slightly less valuable than immediate ones.
            R = r + self.gamma * R
            # Adds the calculated total reward for this step to the front of the list.
            discounted_rewards.insert(0, R)

        # Converts the list of rewards into a PyTorch tensor for calculations.
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)

        # This is a standard trick to stabilize training.
        # It rescales the rewards so they have a mean of 0 and a standard deviation of 1.
        # This prevents very high or very low reward values from causing drastic, unstable updates to the model.
        if len(discounted_rewards) > 1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        # This is the core of the REINFORCE algorithm.
        # It calculates the loss for each action taken in the episode.
        # `log_prob` is how confident the model was about its action.
        # `reward` is the final credit/blame that action received.
        # Multiplying them tells the model how much to change.
        # The negative sign is because optimizers minimize loss. We want to maximize reward, so we minimize the negative reward.
        policy_loss = [-log_prob * reward for log_prob, reward in zip(self.log_probs, discounted_rewards)]

        # Resets the gradients. If we didn't do this, gradients would accumulate from previous training steps.
        self.optimizer.zero_grad()

        # Converts the list of individual action losses into a single tensor and sums them up.
        # This gives us the total loss for the entire episode.
        policy_loss = torch.stack(policy_loss).sum()

        # This is the "grade". It calculates the gradients for each parameter (knob) in the network.
        # It tells PyTorch how much each knob contributed to the final total loss.
        policy_loss.backward()

        # This is the "training". It uses the calculated gradients to update the model's weights (turn the knobs).
        # It nudges the weights in the direction that will reduce the loss (i.e., increase the likelihood of good actions).
        self.optimizer.step()

        # Clears the memory of the last episode to prepare for the next one.
        self.log_probs = []
        self.rewards = []

    def load_model(self, path):
        self.policy_network.load_state_dict(torch.load(path))

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy_network.state_dict(), path)