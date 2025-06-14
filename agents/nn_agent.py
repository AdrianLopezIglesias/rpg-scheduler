import joblib
import numpy as np
import copy
from .base_agent import Agent
import random

def create_feature_vector(game_state, game):
    """Creates the 'Threat-Aware' feature vector from a given game state."""
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

class NNAgent(Agent):
    """The original agent that uses a state-value model to look one step ahead."""
    def __init__(self, model_path_prefix, epsilon=0.1):
        self.epsilon = epsilon
        try:
            self.model = joblib.load(f"{model_path_prefix}.joblib")
            self.scaler = joblib.load(f"{model_path_prefix}_scaler.joblib")
        except FileNotFoundError:
            self.model = None

    def _simulate_next_state(self, game, action):
        """Creates a hypothetical future state for evaluation."""
        sim_game = copy.deepcopy(game)
        # Note: This agent is legacy and will not work with the updated step function.
        sim_game.step(action)
        return sim_game.get_state_snapshot()

    def choose_action(self, game, possible_actions):
        if not self.model:
            return random.choice(possible_actions)
        if random.random() < self.epsilon:
            return random.choice(possible_actions)
        
        best_action = None
        best_predicted_score = -float('inf')
        # Note: This logic is legacy and non-functional with the current game state.
        for action in possible_actions:
            pass
        return best_action if best_action else random.choice(possible_actions)