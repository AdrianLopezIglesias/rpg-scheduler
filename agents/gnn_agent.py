import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.distributions import Categorical
from .base_agent import Agent

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
        # `input_dim` is the number of features per city.
        # `hidden_dim` is the size of the secret summary it creates (e.g., 32 numbers).
        self.conv1 = GCNConv(input_dim, hidden_dim)
        
        # This is the second GNN layer. It takes the summary from the first layer
        # and processes it again, allowing the model to learn even more complex
        # relationships between cities and their neighbors. It helps the model "think" two steps out.
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # --- These are the "specialists" ---
        
        # The 'Move' Specialist: A simple linear layer.
        self.move_head = nn.Linear(hidden_dim, 1)
        
        # The 'Treat' Specialist: Another independent linear layer.
        self.treat_head = nn.Linear(hidden_dim, 1)

        # The 'Cure' Specialist: A head to evaluate discovering a cure for each color.
        self.cure_head = nn.Linear(hidden_dim, 4) # 4 colors
        
        # The 'Pass' Specialist: A third independent linear layer.
        self.pass_head = nn.Linear(hidden_dim, 1)

    # This is the `forward` pass. It defines what happens when we feed the game state into the brain.
    def forward(self, data):
        # Unpacks the graph data object we give it.
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # === Step 1: Create the secret summaries (node embeddings) ===
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        node_embeddings = self.conv2(x, edge_index)
        
        # === Step 2: Get a summary of the whole board ===
        graph_embedding = global_mean_pool(node_embeddings, batch)
        
        # The brain's "thinking" is done. It now returns the processed embeddings.
        return node_embeddings, graph_embedding

# This class represents the agent itself. It holds the "brain" (the PolicyNetwork)
# and manages the process of choosing actions and learning from experience.
class GNNAgent(Agent):
    """RL Agent that uses the map-agnostic GNN."""

    # The constructor for the agent. It runs once when we create the agent.
    def __init__(self, input_dim, config):
        # Gets the specific configuration for the Reinforcement Learning process.
        rl_cfg = config.get('rl_config') or config.get('expert_config')
        
        # Creates the agent's brain.
        self.policy_network = PolicyNetwork(input_dim)
        
        # Creates an "optimizer" to update the weights of our PolicyNetwork.
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=rl_cfg['learning_rate'])
        
        # Gamma determines how much the agent values future rewards.
        self.gamma = rl_cfg['gamma']
        
        # Agent's short-term memory for a single game.
        self.log_probs = []
        self.rewards = []
        self.colors = ["blue", "yellow", "black", "red"]

    def choose_action(self, game, state_graph):
        """
        Calculates scores for only the currently legal actions and chooses one.
        """
        # First, the agent "thinks" by passing the current game state through its brain.
        (node_embeddings, graph_embedding) = self.policy_network(state_graph)
        
        # Ask the game for the list of currently legal moves.
        possible_actions_mask = game.get_possible_action_mask()
        
        # Create a list of scores (logits) and initialize all to a very low number.
        logits = torch.full_like(possible_actions_mask, -1e8, dtype=torch.float)

        # Loop through only the legal actions to get their scores.
        for action_idx, is_possible in enumerate(possible_actions_mask):
            if is_possible:
                action_desc = game.idx_to_action[action_idx]
                score = 0
                action_type = action_desc.get("type")
                
                # Consult the correct "specialist" for the action type.
                if action_type == 'move':
                    score = self.policy_network.move_head(node_embeddings[action_desc['target_idx']])
                elif action_type == 'treat':
                    score = self.policy_network.treat_head(node_embeddings[action_desc['target_idx']])
                elif action_type == 'discover_cure':
                    color_scores = self.policy_network.cure_head(graph_embedding)
                    color_idx = self.colors.index(action_desc['color'])
                    score = color_scores[0, color_idx]
                elif action_type == 'pass':
                    score = self.policy_network.pass_head(graph_embedding)
                
                # Update the score for this legal action.
                logits[action_idx] = score

        # Create a probability distribution from the scores.
        prob_dist = Categorical(logits=logits)
        
        # Sample from the distribution to pick an action.
        chosen_action_idx = prob_dist.sample()
        
        # Save the confidence score (log_prob) for learning later.
        self.log_probs.append(prob_dist.log_prob(chosen_action_idx))
        
        # Return the chosen action.
        return chosen_action_idx.item()

    def update_policy(self):
        # Initializes an empty list to hold the calculated rewards for each step.
        discounted_rewards = []
        # This variable, R, will hold the running total of the reward. It starts at 0.
        R = 0
        # This loop iterates backwards through the raw rewards collected during the episode.
        for r in self.rewards[::-1]:
            # The core of credit assignment.
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)

        # Converts the list of rewards into a PyTorch tensor.
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        # This is a standard trick to stabilize training.
        if len(discounted_rewards) > 1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        # This is the core of the REINFORCE algorithm.
        # It calculates the loss for each action taken.
        policy_loss = [-log_prob * reward for log_prob, reward in zip(self.log_probs, discounted_rewards)]

        # Resets the gradients.
        self.optimizer.zero_grad()
        # Converts the list of losses into a single value.
        policy_loss = torch.stack(policy_loss).sum()
        # This is the "grade". It calculates how to change each parameter.
        policy_loss.backward()
        # This is the "training". It updates the model's weights.
        self.optimizer.step()
        # Clears the memory for the next episode.
        self.log_probs = []
        self.rewards = []

    def load_model(self, path):
        self.policy_network.load_state_dict(torch.load(path))

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy_network.state_dict(), path)