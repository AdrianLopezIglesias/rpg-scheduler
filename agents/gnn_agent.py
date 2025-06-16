import os
import torch
import torch.optim as optim
from torch.distributions import Categorical
from .base_agent import Agent
from .policy_network import PolicyNetwork

class GNNAgent(Agent):
    def __init__(self, input_dim, config):
        rl_cfg = config.get('rl_config')
        self.policy_network = PolicyNetwork(input_dim)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=rl_cfg['learning_rate'])
        self.gamma = rl_cfg['gamma']
        self.log_probs = []
        self.rewards = []
        self.colors = ["blue", "yellow", "black", "red"]

    def choose_action(self, game, state_graph):
        (node_embeddings, graph_embedding) = self.policy_network(state_graph)
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
                    score = self.policy_network.build_head(graph_embedding)
                elif action_type == 'pass':
                    score = self.policy_network.pass_head(graph_embedding)
                
                logits[action_idx] = score

        prob_dist = Categorical(logits=logits)
        chosen_action_idx = prob_dist.sample()
        self.log_probs.append(prob_dist.log_prob(chosen_action_idx))
        return chosen_action_idx.item()

    def update_policy(self):
        # ... (method is unchanged)
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