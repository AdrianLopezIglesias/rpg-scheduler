import os
import json
import joblib
import numpy as np
from datetime import datetime
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from agents.agents import create_feature_vector
from game.pandemic_game import PandemicGame 

# --- Report Configuration ---
# A single timestamp is generated when the module is first loaded.
# This ensures all generations within a single run log to the same report file.
RUN_TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
REPORTS_DIR = 'reports'
os.makedirs(REPORTS_DIR, exist_ok=True)
REPORT_PATH = os.path.join(REPORTS_DIR, f'training_report_{RUN_TIMESTAMP}.json')


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
    # Pass the config to the game instance to ensure it initializes correctly
    game_instance_for_calcs = PandemicGame(difficulty=difficulty, config=config)
    
    for game_data in elite_games:
        score = 1000 - game_data["total_actions"]
        for turn in game_data["game_history"]:
            state = turn['state']
            feature_vector = create_feature_vector(state, game_instance_for_calcs)
            X_list.append(feature_vector[0])
            y_list.append(score)

    return np.array(X_list), np.array(y_list)

def update_report(generation_num, difficulty, analysis_results, config):
    """Loads, updates, and saves a JSON report of the training progress."""
    try:
        # Try to load the report for the current run.
        with open(REPORT_PATH, 'r') as f:
            report = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # If it doesn't exist, create a new one.
        report = {
            "training_run_config": config,
            "generational_results": []
        }

    # Add the current generation's results.
    report_entry = {
        "generation": generation_num,
        "difficulty": difficulty,
        **analysis_results
    }
    report["generational_results"].append(report_entry)

    # Sort results by generation to keep the report clean.
    report["generational_results"].sort(key=lambda r: r['generation'])

    with open(REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"Training report updated: {REPORT_PATH}")

def train_next_generation(current_generation_num, difficulty, config, analysis_results):
    """The main training function, now correctly accepts analysis_results for reporting."""
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

    # After training, update the report with the performance of this generation.
    update_report(current_generation_num, difficulty, analysis_results, config)
