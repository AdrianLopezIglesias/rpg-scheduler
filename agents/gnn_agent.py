import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from .base_agent import Agent
from .policy_network import PolicyNetwork
import random

class GNNAgent(Agent):
    def __init__(self, input_dim, config):
        self.rl_cfg = config.get('rl_config', {})
        self.policy_network = PolicyNetwork(input_dim)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.rl_cfg['learning_rate'])
        self.gamma = self.rl_cfg['gamma']
        
        self.temp_prob = self.rl_cfg.get("temperature_prob", 0.0)
        self.temp_value = self.rl_cfg.get("temperature_value", 2.0)
        self.temp_weights = self.rl_cfg.get("temperature_weights", {})
        self.temp_strategies = list(self.temp_weights.keys())
        self.temp_weight_values = list(self.temp_weights.values())

        self.entropy_coeff = self.rl_cfg.get("entropy_coeff", 0.01)
        
        self.entropies = []
        self.log_probs = []
        self.rewards = []
        self.values = [] # To store critic value predictions
        self.colors = ["blue", "yellow", "black", "red"]

    def choose_action(self, game, state_graph):
        # MODIFIED: Unpack the state_value returned by the Actor-Critic network
        (node_embeddings, graph_embedding, state_value) = self.policy_network(state_graph)
        
        possible_actions_mask = game.get_possible_action_mask()
        logits = torch.full_like(possible_actions_mask, -1e8, dtype=torch.float)

        for action_idx, is_possible in enumerate(possible_actions_mask):
            if is_possible:
                action_desc = game.idx_to_action[action_idx]
                score = 0
                action_type = action_desc.get("type")
                
                if action_type == 'move':
                    target_node_embedding = node_embeddings[action_desc['target_idx']]
                    combined_embedding = torch.cat([target_node_embedding, graph_embedding.squeeze(0)])
                    score = self.policy_network.move_head(combined_embedding)
                elif action_type == 'treat':
                    target_node_embedding = node_embeddings[action_desc['target_idx']]
                    combined_embedding = torch.cat([target_node_embedding, graph_embedding.squeeze(0)])
                    score = self.policy_network.treat_head(combined_embedding)
                elif action_type == 'discover_cure':
                    color_scores = self.policy_network.cure_head(graph_embedding)
                    color_idx = self.colors.index(action_desc['color'])
                    score = color_scores[0, color_idx]
                elif action_type == 'build_investigation_center':
                    target_node_embedding = node_embeddings[action_desc['target_idx']]
                    combined_embedding = torch.cat([target_node_embedding, graph_embedding.squeeze(0)])
                    score = self.policy_network.build_head(combined_embedding)
                elif action_type == 'pass':
                    score = self.policy_network.pass_head(graph_embedding)
                
                logits[action_idx] = score
        
        chosen_action_idx_int = -1
        if self.temp_prob > 0 and sum(self.temp_weight_values) > 0 and random.random() < self.temp_prob:
            chosen_strategy = random.choices(self.temp_strategies, weights=self.temp_weight_values, k=1)[0]
            
            possible_indices = [i for i, p in enumerate(possible_actions_mask) if p]
            if possible_indices:
                if chosen_strategy == "full_random":
                    chosen_action_idx_int = random.choice(possible_indices)
                elif chosen_strategy == "top_2_random":
                    possible_logits = {i: logits[i].item() for i in possible_indices}
                    top_2_indices = sorted(possible_logits, key=possible_logits.get, reverse=True)[:2]
                    chosen_action_idx_int = random.choice(top_2_indices)
                elif chosen_strategy == "fixed_bonus":
                    bonus_idx = random.choice(possible_indices)
                    logits[bonus_idx] += self.temp_value
                elif chosen_strategy == "proportional_bonus":
                    bonus_idx = random.choice(possible_indices)
                    logits[bonus_idx] += random.uniform(0, self.temp_value)

        prob_dist = Categorical(logits=logits)
        self.entropies.append(prob_dist.entropy())

        if chosen_action_idx_int != -1:
            chosen_action_idx = torch.tensor(chosen_action_idx_int)
        else:
            chosen_action_idx = prob_dist.sample()
        
        self.log_probs.append(prob_dist.log_prob(chosen_action_idx))
        self.values.append(state_value) # Store the predicted state value

        return chosen_action_idx.item()

    # In agents/gnn_agent.py

    def update_policy(self):
        # REWRITTEN: Actor-Critic Update Logic
        returns = []
        R = 0
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        log_probs = torch.stack(self.log_probs)
        values = torch.stack(self.values).squeeze(-1)
        entropies = torch.stack(self.entropies)
        
        # --- KEY CHANGES ARE HERE ---
        advantage = returns - values.detach() # Detach values for advantage calculation
        actor_loss = -(log_probs * advantage).mean() # No longer need .detach() here
        
        critic_loss = F.smooth_l1_loss(values, returns)
        entropy_loss = -self.entropy_coeff * entropies.mean()
        loss = actor_loss + critic_loss + entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_probs = []
        self.rewards = []
        self.values = []
        self.entropies = []

    def load_model(self, path):
        self.policy_network.load_state_dict(torch.load(path))
        
    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy_network.state_dict(), path)