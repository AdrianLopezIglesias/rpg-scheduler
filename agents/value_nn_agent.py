import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import json
import random

from .base_agent import Agent

class ValueNet(nn.Module):
    """A simple MLP to evaluate a game state vector."""
    def __init__(self, input_dim, hidden_dim1=256, hidden_dim2=128):
        super(ValueNet, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim1)
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.layer3 = nn.Linear(hidden_dim2, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class ValueNNAgent(Agent):
    """An agent that uses a ValueNet to look one step ahead."""
    def __init__(self, model_path, input_dim, config, device='cpu'):
        self.device = device
        self.model = ValueNet(input_dim).to(self.device)
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            except FileNotFoundError:
                print(f"Warning: Model file not found at {model_path}. Using a randomly initialized model.")
        self.model.eval()

    def _simulate_and_get_vector(self, game, action_idx):
        """Simulates a move and returns the resulting state vector."""
        sim_game = copy.deepcopy(game)
        sim_game.step(action_idx)
        return sim_game.get_state_as_vector()

    def choose_action(self, game, possible_actions_mask):
        """
        Evaluates all possible actions by simulating them and scoring the
        resulting state with the ValueNet.
        """
        best_action_idx = -1
        best_score = -float('inf')

        possible_indices = [i for i, p in enumerate(possible_actions_mask) if p]
        if not possible_indices:
            # Should not happen if a 'pass' action is always available
            return game.action_to_idx[json.dumps({"type": "pass"})]

        with torch.no_grad():
            for action_idx in possible_indices:
                # Simulate the action to get the next state vector
                next_state_vector = self._simulate_and_get_vector(game, action_idx)
                state_tensor = torch.tensor(next_state_vector, dtype=torch.float32).to(self.device)

                # Get the score from the critic
                predicted_score = self.model(state_tensor).item()

                if predicted_score > best_score:
                    best_score = predicted_score
                    best_action_idx = action_idx
        
        # Fallback to a random action if no best action is found
        return best_action_idx if best_action_idx != -1 else random.choice(possible_indices)

