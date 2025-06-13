import os
import json
import joblib
import numpy as np
import shutil
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from agents.agents import create_feature_vector
from game.pandemic_game import PandemicGame 

def load_data(filepath):
    try:
        with open(filepath, "r") as f: return json.load(f)
    except FileNotFoundError: return None

def get_elite_games(list_of_games, num_to_keep):
    if not list_of_games: return []
    winning_games = [g for g in list_of_games if g.get("final_result") == "win"]
    winning_games.sort(key=lambda g: g["total_actions"])
    return winning_games[:num_to_keep]

def preprocess_training_data(list_of_games, difficulty, config):
    X_list, y_list = [], []
    game_instance = PandemicGame(difficulty=difficulty, config=config)
    
    for game_data in list_of_games:
        score = 1000 - game_data["total_actions"]
        for turn in game_data["game_history"]:
            state = turn['state']
            feature_vector = create_feature_vector(state, game_instance)[0]
            X_list.append(feature_vector)
            y_list.append(score)

    return np.array(X_list), np.array(y_list)

def train_model_on_data(current_generation_num, difficulty, config, output_path_prefix):
    model_cfg = config['model_config']
    trainer_cfg = config['trainer_config']
    
    all_historical_games = []
    for i in range(current_generation_num + 1):
        data_path = f"data/{difficulty}/generation_{i}/simulation_data.json"
        if os.path.exists(data_path):
            sim_data = load_data(data_path)
            if sim_data: all_historical_games.extend(sim_data)

    if not all_historical_games: return
        
    elite_games = get_elite_games(all_historical_games, trainer_cfg['elite_games_to_keep'])
    if not elite_games:
        print("No winning games found in historical data. Cannot train.")
        return
    
    print(f"Training on data from {len(elite_games)} best historical games.")
    total_turns = sum(len(g['game_history']) for g in elite_games)
    print(f"Total training samples (turns): {total_turns}")
    
    X, y = preprocess_training_data(elite_games, difficulty, config)
    if X.shape[0] < 2: return

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    
    model = MLPRegressor(random_state=42, hidden_layer_sizes=model_cfg['hidden_layer_sizes'], alpha=model_cfg['alpha'], max_iter=2000)
    model.fit(X_scaled, y)
        
    output_dir = os.path.dirname(output_path_prefix)
    os.makedirs(output_dir, exist_ok=True)
    
    joblib.dump(model, f"{output_path_prefix}.joblib")
    joblib.dump(scaler, f"{output_path_prefix}_scaler.joblib")
    print(f"New candidate model saved to {output_path_prefix}.joblib")
