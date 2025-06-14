import os
import torch
import torch.optim as optim
from torch.distributions import Categorical

from .base_agent import Agent
# It now imports its 'brain' from the new file.
from .policy_network import PolicyNetwork

class GNNAgent(Agent):
    """The 'President' agent that uses the PolicyNetwork to act and learn."""
    def __init__(self, input_dim, config):
        rl_cfg = config.get('rl_config') or config.get('expert_config')
        
        # The President's direct line to the Intelligence Agency.
        self.policy_network = PolicyNetwork(input_dim)
        
        # The tool the President uses to enact policy changes (update the brain).
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=rl_cfg['learning_rate'])
        
        self.gamma = rl_cfg['gamma']
        self.log_probs = []
        self.rewards = []
        self.colors = ["blue", "yellow", "black", "red"]

    def choose_action(self, game, state_graph):
        """The 'Cabinet Meeting' where the President makes a decision."""
        # The President requests a full briefing from the Intelligence Agency.
        (node_embeddings, graph_embedding) = self.policy_network(state_graph)
        
        # The President only considers currently possible executive actions.
        possible_actions_mask = game.get_possible_action_mask()
        
        # A list to hold the Approval Score for each possible action.
        logits = torch.full_like(possible_actions_mask, -1e8, dtype=torch.float)

        # The President consults the Cabinet for each legal action.
        for action_idx, is_possible in enumerate(possible_actions_mask):
            if is_possible:
                action_desc = game.idx_to_action[action_idx]
                score = 0
                action_type = action_desc.get("type")
                
                # Consult the correct Secretary (specialist head).
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
                
                logits[action_idx] = score

        # The President makes a choice based on the Cabinet's scores.
        prob_dist = Categorical(logits=logits)
        chosen_action_idx = prob_dist.sample()
        
        # The President's Press Secretary records the decision for the history books.
        self.log_probs.append(prob_dist.log_prob(chosen_action_idx))
        
        return chosen_action_idx.item()

    def update_policy(self):
        """The 'Judgment of History' where the Historians update the official record."""
        # Calculate the legacy of each decision (discounted rewards).
        discounted_rewards = []
        R = 0
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)

        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        if len(discounted_rewards) > 1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        # Calculate the historical impact (policy loss).
        policy_loss = [-log_prob * reward for log_prob, reward in zip(self.log_probs, discounted_rewards)]

        # The Historians present their findings to enact change.
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        # The record is cleared for the next term.
        self.log_probs = []
        self.rewards = []

    def load_model(self, path):
        """Load a previous administration's 'brain'."""
        self.policy_network.load_state_dict(torch.load(path))

    def save_model(self, path):
        """Archive the current administration's 'brain'."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy_network.state_dict(), path)