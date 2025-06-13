import random
import json
import joblib
import numpy as np
import copy

def create_feature_vector(game_state, game):
    """Creates the 'Threat-Aware' feature vector from a given game state."""
    features = []
    player_loc = game_state['player_location']
    board = game_state['board']
    
    features.append(board[player_loc]['cubes'])
    neighbors1 = sorted(game.map[player_loc]['neighbors'])
    for i in range(4): # This can stay at 4 as it's part of the feature vector's fixed size
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
        return random.choice(possible_actions)

class NNAgent(Agent):
    """An agent that uses a model to predict a score for each possible generic action."""
    def __init__(self, model_path_prefix, epsilon=0.1):
        self.epsilon = epsilon
        try:
            self.model = joblib.load(f"{model_path_prefix}.joblib")
            self.scaler = joblib.load(f"{model_path_prefix}_scaler.joblib")
        except FileNotFoundError:
            self.model = None

    def choose_action(self, game, possible_actions):
        if not self.model:
            return random.choice(possible_actions)

        if random.random() < self.epsilon:
            return random.choice(possible_actions)

        feature_vector = create_feature_vector(game.get_state_snapshot(), game)
        scaled_features = self.scaler.transform(feature_vector)
        action_scores = self.model.predict(scaled_features)[0]

        best_action = None
        best_score = -float('inf')
        
        # Action 0: Treat
        action_treat = {"type": "treat", "target": game.player_location}
        if action_treat in possible_actions and action_scores[0] > best_score:
            best_score = action_scores[0]
            best_action = action_treat
            
        # --- FIX: Check all sorted neighbors, not just the first 4 ---
        sorted_neighbors = sorted(game.map[game.player_location]['neighbors'])
        for i, neighbor in enumerate(sorted_neighbors):
            # The model's output for moves starts at index 1
            action_score_index = i + 1
            # Ensure we don't go out of bounds of the model's output layer
            if action_score_index < len(action_scores):
                score_move = action_scores[action_score_index]
                action_move = {"type": "move", "target": neighbor}
                if action_move in possible_actions and score_move > best_score:
                    best_score = score_move
                    best_action = action_move

        return best_action if best_action else random.choice(possible_actions)
