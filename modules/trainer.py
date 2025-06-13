import os
import json
import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from agents.agents import create_feature_vector
from game.pandemic_game import PandemicGame 

def load_data(filepath):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def preprocess_training_data(list_of_games, difficulty, config):
    cfg = config['trainer_config']
    winning_games = [g for g in list_of_games if g.get("final_result") == "win"]
    if not winning_games:
        print("Warning: No winning games found. Cannot train.")
        return None, None
        
    winning_games.sort(key=lambda g: g["total_actions"])
    elite_games = winning_games[:cfg['elite_games_to_keep']]
    
    print(f"Training on data from {len(elite_games)} best games.")

    training_data = [turn for game in elite_games for turn in game["game_history"]]
    
    print(f"Total training samples (turns): {len(training_data)}")

    X_list, y_list = [], []
    game_instance_for_calcs = PandemicGame(difficulty=difficulty)
    
    for game_data in elite_games:
        score = 1000 - game_data["total_actions"]
        for turn in game_data["game_history"]:
            state = turn['state']
            feature_vector = create_feature_vector(state, game_instance_for_calcs)
            X_list.append(feature_vector[0])
            y_list.append(score)

    return np.array(X_list), np.array(y_list)

def train_next_generation(current_generation_num, difficulty, config):
    model_cfg = config['model_config']
    all_historical_games = []
    
    data_dir = f"data/{difficulty}"
    for i in range(current_generation_num + 1):
        data_path = f"{data_dir}/generation_{i}/simulation_data.json"
        if os.path.exists(data_path):
            sim_data = load_data(data_path)
            if sim_data:
                all_historical_games.extend(sim_data)
    
    if not all_historical_games:
        return

    X, y = preprocess_training_data(all_historical_games, difficulty, config)
    
    if X is None or X.shape[0] < 2: return

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    
    model = MLPRegressor(
        random_state=42, 
        max_iter=1000, 
        hidden_layer_sizes=model_cfg['hidden_layer_sizes'], 
        alpha=model_cfg['alpha'],
        verbose=False
    )
    model.fit(X_scaled, y) 

    output_dir = f"models/{difficulty}/generation_{current_generation_num + 1}"
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, f"{output_dir}/pandemic_model.joblib")
    joblib.dump(scaler, f"{output_dir}/pandemic_model_scaler.joblib")
