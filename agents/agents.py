import random
import json
import joblib
import numpy as np
import os

# A helper function that might be needed by the NN Agent
def flatten_state_for_nn(state, city_encoder):
    """Converts the nested state dictionary into a flat numerical vector for the NN."""
    board_state = state['board']
    cube_features = [board_state.get(city, {}).get('cubes', 0) for city in city_encoder.classes_]
    player_location_encoded = city_encoder.transform([state['player_location']])[0]
    features = cube_features + [player_location_encoded, state['current_round'], state['actions_remaining']]
    return np.array(features).reshape(1, -1) # Reshape for single prediction


class Agent:
    """Base class for all agents."""
    def choose_action(self, game_state, possible_actions):
        raise NotImplementedError

class RandomAgent(Agent):
    """An agent that chooses its action randomly from the list of possible actions."""
    def choose_action(self, game_state, possible_actions):
        return random.choice(possible_actions)

class NNAgent(Agent):
    """An agent that uses a trained Neural Network to choose an action."""
    def __init__(self, model_path_prefix):
        """
        Loads a trained model and its associated transformers.
        Example prefix: 'models/generation_1/pandemic_model'
        """
        try:
            self.model = joblib.load(f"{model_path_prefix}.joblib")
            self.scaler = joblib.load(f"{model_path_prefix}_scaler.joblib")
            self.city_encoder = joblib.load(f"{model_path_prefix}_city_encoder.joblib")
            self.action_encoder = joblib.load(f"{model_path_prefix}_action_encoder.joblib")
        except FileNotFoundError:
            print(f"Error: Could not load model from prefix {model_path_prefix}")
            self.model = None

    def choose_action(self, game_state, possible_actions):
        if not self.model:
            print("NN Agent has no model loaded, choosing randomly.")
            return random.choice(possible_actions)

        # 1. Preprocess the current game state
        flat_state = flatten_state_for_nn(game_state, self.city_encoder)
        scaled_state = self.scaler.transform(flat_state)
        
        # 2. Get action probabilities from the model
        # The model outputs a probability for every possible action it was trained on.
        action_probabilities = self.model.predict_proba(scaled_state)[0]
        
        # 3. Find the best VALID action
        # We can't just pick the highest probability action, because it might not be possible
        # in the current game state (e.g., moving to a non-adjacent city).
        best_action = None
        max_prob = -1

        # Iterate through the model's learned actions and their probabilities
        for i, action_str in enumerate(self.action_encoder.classes_):
            action_dict = json.loads(action_str)
            
            # Check if this action is in our list of currently possible actions
            if action_dict in possible_actions and action_probabilities[i] > max_prob:
                max_prob = action_probabilities[i]
                best_action = action_dict
        
        # Fallback to random choice if no valid action was found (should be rare)
        return best_action if best_action else random.choice(possible_actions)

