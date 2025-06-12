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
        """
        An optimized method to choose an action by only evaluating the probabilities
        of currently valid moves.
        """
        if not self.model:
            return random.choice(possible_actions)

        # 1. Preprocess the current game state
        flat_state = flatten_state_for_nn(game_state, self.city_encoder)
        scaled_state = self.scaler.transform(flat_state)
        
        # 2. Get action probabilities for ALL known actions just once
        action_probabilities = self.model.predict_proba(scaled_state)[0]
        
        # 3. Create a fast lookup map of {encoded_action: probability}
        # This maps the integer label of an action to its predicted probability.
        prob_map = {action: prob for action, prob in zip(self.model.classes_, action_probabilities)}

        # 4. Find the best VALID action efficiently
        best_action = None
        max_prob = -1
        
        # Iterate through only the few currently possible actions
        for action in possible_actions:
            action_str = json.dumps(action, sort_keys=True)
            
            # Check if this action is one the model was trained on
            if action_str in self.action_encoder.classes_:
                encoded_action = self.action_encoder.transform([action_str])[0]
                
                # Look up the probability for this specific action
                prob = prob_map.get(encoded_action, -1)

                if prob > max_prob:
                    max_prob = prob
                    best_action = action

        # Fallback to random if no known valid action is found
        return best_action if best_action else random.choice(possible_actions)
