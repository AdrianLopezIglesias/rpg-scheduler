import os
import json
import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from agents.agents import create_feature_vector
from game.pandemic_game import PandemicGame 

def load_data(filepath):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def get_all_cities(difficulty="hard"):
    config_path = os.path.join(os.path.dirname(__file__), 'game', 'maps.json')
    with open(config_path, 'r') as f:
        all_maps = json.load(f)
    return list(all_maps[difficulty]["cities"].keys())

def map_action_to_output_index(action, state, game):
    """Maps a specific action to its corresponding generic output index."""
    if action['type'] == 'treat':
        return 0
    if action['type'] == 'move':
        sorted_neighbors = sorted(game.map[state['player_location']]['neighbors'])
        try:
            return sorted_neighbors.index(action['target']) + 1
        except ValueError:
            return -1
    return -1

def preprocess_training_data(list_of_games, difficulty):
    print("Preprocessing data for action-scoring model...")
    winning_games = [g for g in list_of_games if g.get("final_result") == "win"]
    if not winning_games:
        print("Warning: No winning games found. Cannot train.")
        return None, None
        
    winning_games.sort(key=lambda g: g["total_actions"])
    elite_games = winning_games[:100]
    
    X_list, y_list = [], []
    game_instance_for_calcs = PandemicGame(difficulty=difficulty)
    
    # --- FIX: Increase model output size to handle more neighbors ---
    # The model will now predict a score for 'treat' + up to 6 moves.
    num_outputs = 7 

    for game_data in elite_games:
        score = 1000 - game_data["total_actions"]
        for turn in game_data["game_history"]:
            state = turn['state']
            action = turn['action_taken']

            feature_vector = create_feature_vector(state, game_instance_for_calcs)
            X_list.append(feature_vector[0])

            target_vector = np.full(num_outputs, -500.0)
            action_index = map_action_to_output_index(action, state, game_instance_for_calcs)

            if action_index != -1 and action_index < num_outputs:
                target_vector[action_index] = score

            y_list.append(target_vector)

    return np.array(X_list), np.array(y_list)


def train_next_generation(current_generation_num, difficulty="hard"):
    all_historical_games = []
    for i in range(current_generation_num + 1):
        data_path = f"data/{difficulty}/generation_{i}/simulation_data.json"
        sim_data = load_data(data_path)
        if sim_data:
            all_historical_games.extend(sim_data)
    
    if not all_historical_games:
        return

    X, y = preprocess_training_data(all_historical_games, difficulty)
    
    if X is None or X.shape[0] < 2:
        return

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    
    model = MLPRegressor(random_state=42, max_iter=1000, hidden_layer_sizes=(100, 50), alpha=0.001)
    print(f"\nTraining action-scoring model for Gen {current_generation_num + 1}...")
    model.fit(X_scaled, y) 

    output_dir = f"models/{difficulty}/generation_{current_generation_num + 1}"
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, f"{output_dir}/pandemic_model.joblib")
    joblib.dump(scaler, f"{output_dir}/pandemic_model_scaler.joblib")
    print(f"New action-scoring model saved for Gen {current_generation_num + 1}")

if __name__ == "__main__":
    GENERATION_TO_TRAIN_ON = 0
    DIFFICULTY = "hard"
    train_next_generation(GENERATION_TO_TRAIN_ON, difficulty=DIFFICULTY)
