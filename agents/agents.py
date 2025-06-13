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
# This defines the blueprint for our agent's brain.
# It inherits from `nn.Module`, which is the base class for all neural network models in PyTorch.
class PolicyNetwork(nn.Module):
    """GNN that learns node embeddings and uses heads to score action types."""
    
    # This is the constructor. It runs when we first create the brain.
    # It sets up all the layers (the "knobs" and "processors") the network will use.
    def __init__(self, input_dim, hidden_dim=32):
        # This line is required for all PyTorch models. It properly initializes the nn.Module parent class.
        super(PolicyNetwork, self).__init__()
        
        # This creates the first Graph Convolutional Network layer.
        # It's the main processor for the graph. It takes the initial, simple features of each city
        # (like cube count and player location) and starts creating a more complex, "hidden" summary.
        # `input_dim` is the number of features per city (e.g., 2: cubes, player_is_here).
        # `hidden_dim` is the size of the secret summary it creates (e.g., 32 numbers).
        self.conv1 = GCNConv(input_dim, hidden_dim)
        
        # This is the second GNN layer. It takes the summary from the first layer
        # and processes it again, allowing the model to learn even more complex
        # relationships between cities and their neighbors. It helps the model "think" two steps out.
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # --- These are the "specialists" ---
        
        # The 'Move' Specialist: A simple linear layer.
        # Its only job is to look at a city's final secret summary (size 32)
        # and output a single number (score) saying "How good is it to MOVE to this city?"
        self.move_head = nn.Linear(hidden_dim, 1)
        
        # The 'Treat' Specialist: Another independent linear layer.
        # Its job is to look at a city's summary and answer: "How good is it to TREAT this city?"
        self.treat_head = nn.Linear(hidden_dim, 1)
        
        # The 'Pass' Specialist: A third independent linear layer.
        # Its job is different. It looks at a summary of the WHOLE board and answers:
        # "How good is it to PASS my turn right now?"
        self.pass_head = nn.Linear(hidden_dim, 1)

    # This is the `forward` pass. It defines what happens when we feed the game state into the brain.
    # It's the step-by-step process of thinking.
    def forward(self, data):
        # Unpacks the graph data object we give it.
        # `x` is the list of city features.
        # `edge_index` defines the map connections.
        # `batch` is used by PyTorch Geometric to handle multiple graphs at once (we only use one).
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # === Step 1: Create the secret summaries (node embeddings) ===
        
        # Pass the city features and map connections through the first GNN layer.
        # This is "message passing" - each city learns from its direct neighbors.
        x = self.conv1(x, edge_index)
        # Apply a ReLU activation function. This is a standard step that helps the network learn
        # complex patterns by adding non-linearity. It's like letting the brain think in curves, not just straight lines.
        x = F.relu(x)
        # Pass the results through the second GNN layer for more sophisticated thinking.
        # Now each city has learned from its neighbors, and its neighbors' neighbors.
        node_embeddings = self.conv2(x, edge_index)
        
        # === Step 2: Get a summary of the whole board ===
        
        # Take the secret summary of every single city and average them together.
        # This creates one single summary vector that represents the entire state of the board.
        graph_embedding = global_mean_pool(node_embeddings, batch)
        
        # The brain's "thinking" is done. It now returns two things:
        # 1. A detailed list of secret summaries for every city (`node_embeddings`).
        # 2. One single summary for the whole game (`graph_embedding`).
        # The agent will use these to ask the specialists for their opinions.
        return node_embeddings, graph_embedding

# --- MODIFIED: GNNAgent now uses the agnostic network ---
# This class represents the agent itself. It holds the "brain" (the PolicyNetwork)
# and manages the process of choosing actions and learning from experience.
class GNNAgent(Agent):
    """RL Agent that uses the map-agnostic GNN."""

    # The constructor for the agent. It runs once when we create the agent.
    def __init__(self, input_dim, config):
        # Gets the specific configuration for the Reinforcement Learning process.
        rl_cfg = config['rl_config']
        
        # Creates the agent's brain by making a new instance of our PolicyNetwork.
        # `input_dim` tells the brain how many features each city has (e.g., 2).
        self.policy_network = PolicyNetwork(input_dim)
        
        # Creates an "optimizer". This is the tool that will physically "turn the knobs"
        # or update the weights of our PolicyNetwork during learning.
        # We tell it which knobs to turn (`self.policy_network.parameters()`) and how fast (`lr`).
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=rl_cfg['learning_rate'])
        
        # Gamma is the "discount factor". It's a number slightly less than 1 (e.g., 0.99)
        # that determines how much the agent values future rewards over immediate ones.
        self.gamma = rl_cfg['gamma']
        
        # These are the agent's short-term memory for a single game (episode).
        # `log_probs` will store how confident the agent was about each action it took.
        self.log_probs = []
        # `rewards` will store the raw reward received at each step (usually all 0 until the end).
        self.rewards = []

    def choose_action(self, game, state_graph):
        """
        Calculates scores for only the currently legal actions and chooses one.
        This is the core of the map-agnostic approach.
        """
        # First, the agent "thinks" by passing the current game state graph through its brain.
        # It gets back the two key outputs: the secret summary for each city (`node_embeddings`)
        # and the single summary for the whole board (`graph_embedding`).
        (node_embeddings, graph_embedding) = self.policy_network(state_graph)
        
        # The agent asks the game, "What are the legal moves I can make right now?"
        # This returns a boolean list, e.g., [False, True, False, True, ...],
        # where True means the action at that index is possible.
        possible_actions_mask = game.get_possible_action_mask()
        
        # The agent creates a list of scores (called "logits") for every single action in the game.
        # It starts by giving every action a very, very bad score (-1e8, which is like negative infinity).
        # This ensures that illegal moves have no chance of being chosen.
        logits = torch.full_like(possible_actions_mask, -1e8, dtype=torch.float)

        # Now, the agent loops through the list of all possible actions in the entire game.
        for action_idx, is_possible in enumerate(possible_actions_mask):
            # It only does work for the actions that are currently legal.
            if is_possible:
                # It gets the description of the legal action (e.g., {type: "move", target_idx: 4}).
                action_desc = game.idx_to_action[action_idx]
                score = 0
                
                # It checks the type of action to decide which "specialist" to consult.
                if action_desc['type'] == 'move':
                    # If it's a move, it gets the secret summary of the target city...
                    target_node_idx = action_desc['target_idx']
                    # ...and gives it to the 'Move' specialist to get a score.
                    score = self.policy_network.move_head(node_embeddings[target_node_idx])
                
                elif action_desc['type'] == 'treat':
                    # If it's a treat, it gets the secret summary of the target city...
                    target_node_idx = action_desc['target_idx']
                    # ...and gives it to the 'Treat' specialist.
                    score = self.policy_network.treat_head(node_embeddings[target_node_idx])
                
                elif action_desc['type'] == 'pass':
                    # If it's a pass, it uses the summary of the whole board...
                    # ...and gives it to the 'Pass' specialist.
                    score = self.policy_network.pass_head(graph_embedding)
                
                # The agent updates the score for this legal action in its list of logits.
                logits[action_idx] = score

        # After scoring all legal moves, the agent creates a probability distribution.
        # Actions with higher scores will have a higher probability of being picked.
        prob_dist = Categorical(logits=logits)
        
        # The agent then samples from this distribution to pick one action.
        # This means it usually picks the highest-scored action, but sometimes
        # it will "explore" by picking a lower-scored one.
        chosen_action_idx = prob_dist.sample()
        
        # The agent saves the log-probability of the action it chose.
        # This number represents how "confident" it was. It's crucial for learning later.
        self.log_probs.append(prob_dist.log_prob(chosen_action_idx))
        
        # Finally, it returns the index of the single action it decided to take.
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