import random
import json
import joblib
import numpy as np
import os

def flatten_state_for_nn(state, city_encoder):
    """Converts the nested state dictionary into a flat numerical vector for the NN."""
    board_state = state['board']
    cube_features = [board_state.get(city, {}).get('cubes', 0) for city in city_encoder.classes_]
    player_location_encoded = city_encoder.transform([state['player_location']])[0]
    
    # FIX: Use 'actions_taken' which exists in the state, instead of the old keys.
    features = cube_features + [player_location_encoded, state['actions_taken']]
    return np.array(features).reshape(1, -1)


class Agent:
    """Base class for all agents."""
    def choose_action(self, game_state, possible_actions):
        raise NotImplementedError

class RandomAgent(Agent):
    """An agent that chooses its action randomly."""
    def choose_action(self, game_state, possible_actions):
        return random.choice(possible_actions)

class NNAgent(Agent):
    """An agent that uses a trained Neural Network to choose an action."""
    def __init__(self, model_path_prefix):
        """Loads a trained model and its associated transformers."""
        try:
            self.model = joblib.load(f"{model_path_prefix}.joblib")
            self.scaler = joblib.load(f"{model_path_prefix}_scaler.joblib")
            self.city_encoder = joblib.load(f"{model_path_prefix}_city_encoder.joblib")
            self.action_encoder = joblib.load(f"{model_path_prefix}_action_encoder.joblib")
        except FileNotFoundError as e:
            print(f"Error: Could not load model from {model_path_prefix}. File not found: {e.filename}")
            self.model = None

    def choose_action(self, game_state, possible_actions):
        if not self.model:
            return random.choice(possible_actions)

        flat_state = flatten_state_for_nn(game_state, self.city_encoder)
        scaled_state = self.scaler.transform(flat_state)
        action_probabilities = self.model.predict_proba(scaled_state)[0]
        
        best_action = None
        max_prob = -1

        for i, encoded_action in enumerate(self.model.classes_):
            action_str = self.action_encoder.inverse_transform([encoded_action])[0]
            action_dict = json.loads(action_str)
            
            if action_dict in possible_actions:
                if action_probabilities[i] > max_prob:
                    max_prob = action_probabilities[i]
                    best_action = action_dict
        
        return best_action if best_action else random.choice(possible_actions)
