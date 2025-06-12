import os
import json
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from agents.agents import flatten_state_for_nn

def load_data(filepath):
    """Loads simulation data from a specified JSON file."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Data file not found at {filepath}")
        return None

def get_all_cities():
    """Helper to get a consistent list of all cities for encoding."""
    # FIX: This now correctly lists the cities from the smaller map.
    return ["Atlanta", "Chicago", "Washington", "Montreal", "New York"]

def preprocess_training_data(list_of_games, all_cities, percentile_to_keep=0.2):
    """
    Processes data by selecting only the actions from the top-performing games.
    """
    print("Preprocessing training data...")
    
    # 1. Filter for only winning games
    winning_games = [g for g in list_of_games if g.get("final_result") == "win"]
    if not winning_games:
        print("Warning: No winning games found in this dataset. Cannot train.")
        return None, None, None, None

    # 2. Sort winning games by speed (fewer actions is better)
    winning_games.sort(key=lambda g: g["total_actions"])

    # 3. Keep only the best games (e.g., top 20%)
    num_to_keep = int(len(winning_games) * percentile_to_keep)
    if num_to_keep == 0 and len(winning_games) > 0:
        num_to_keep = 1 # Always keep at least one game if possible
    
    elite_games = winning_games[:num_to_keep]
    print(f"Found {len(winning_games)} winning games. Training on the best {len(elite_games)}.")

    # 4. Extract {state, action} pairs from these elite games
    training_data = [turn for game in elite_games for turn in game["game_history"]]

    X_list, y_list = [], []
    city_encoder = LabelEncoder().fit(all_cities)
    
    # Ensure all possible actions are learned by the encoder
    all_possible_actions = set()
    for game in list_of_games:
        for turn in game['game_history']:
            all_possible_actions.add(json.dumps(turn['action_taken'], sort_keys=True))

    action_encoder = LabelEncoder().fit(list(all_possible_actions))

    for turn in training_data:
        state_features = flatten_state_for_nn(turn['state'], city_encoder)
        X_list.append(state_features[0])
        action_str = json.dumps(turn['action_taken'], sort_keys=True)
        y_list.append(action_encoder.transform([action_str])[0])

    return np.array(X_list), np.array(y_list), city_encoder, action_encoder


def train_next_generation(current_generation_num):
    """Loads all historical data, trains a new model, and saves it."""
    all_historical_games = []
    print("Loading all historical data...")
    for i in range(current_generation_num + 1):
        data_path = f"data/generation_{i}/simulation_data.json"
        sim_data = load_data(data_path)
        if sim_data:
            all_historical_games.extend(sim_data)
    
    if not all_historical_games:
        print("No historical data found to train on.")
        return

    all_cities = get_all_cities()
    training_result = preprocess_training_data(all_historical_games, all_cities)
    if not training_result[0] is not None:
        return

    X, y, city_enc, action_enc = training_result
    
    if X.shape[0] < 2: # Need at least 2 samples to train/split
        print("Not enough valid data available for training.")
        return

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    
    model = MLPClassifier(random_state=42, max_iter=1000, hidden_layer_sizes=(50, 25), alpha=0.001, verbose=False)
    print(f"\nTraining model for Generation {current_generation_num + 1} on {X.shape[0]} elite action samples...")
    model.fit(X_scaled, y) 

    output_dir = f"models/generation_{current_generation_num + 1}"
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, f"{output_dir}/pandemic_model.joblib")
    joblib.dump(scaler, f"{output_dir}/pandemic_model_scaler.joblib")
    joblib.dump(city_enc, f"{output_dir}/pandemic_model_city_encoder.joblib")
    joblib.dump(action_enc, f"{output_dir}/pandemic_model_action_encoder.joblib")
    print(f"New model saved for Generation {current_generation_num + 1} in '{output_dir}'")

if __name__ == "__main__":
    GENERATION_TO_TRAIN_ON = 0
    train_next_generation(GENERATION_TO_TRAIN_ON)
