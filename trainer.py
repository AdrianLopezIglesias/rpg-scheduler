# --- Intel Extension for Scikit-learn ---
# This patch can significantly speed up training on Intel CPUs.
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    print("Scikit-learn has been patched with Intel Extension for faster performance.")
except ImportError:
    print("Intel Extension for Scikit-learn not found. Running with standard Scikit-learn.")
    pass

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from agents.agents import flatten_state_for_nn # Re-using the helper function

def load_data(filepath):
    """Loads simulation data from a specified JSON file."""
    try:
        with open(filepath, "r") as f:
            # The data is a list of games, where each game is a list of turns.
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Data file not found at {filepath}")
        return None

def get_all_cities():
    """Helper to get a consistent list of all cities."""
    # In a real project, this would come from a shared config file.
    return [
        "San Francisco", "Chicago", "Atlanta", "Montreal", "Washington", 
        "New York", "Los Angeles", "Mexico City", "Miami", "Tokyo", 
        "Manila", "London", "Madrid", "Sydney", "Lima", "Bogota"
    ]

def preprocess_training_data(list_of_games, all_cities):
    """Processes simulation data into features, labels, and weights."""
    print("Preprocessing training data...")
    
    # Flatten the list of games into a single list of all turns.
    all_turns = [turn for game in list_of_games for turn in game]
    
    # Filter for only the best games. Here, we'll take turns from games with a positive fitness score.
    high_fitness_data = [d for d in all_turns if d.get("final_fitness", 0) > 0]
    if not high_fitness_data:
        print("Warning: No games with positive fitness scores found. Using all game data.")
        high_fitness_data = all_turns

    X_list, y_list, weights = [], [], []

    # Prepare encoders
    city_encoder = LabelEncoder().fit(all_cities)
    unique_actions = {json.dumps(d['action_taken'], sort_keys=True) for d in high_fitness_data}
    action_encoder = LabelEncoder().fit(list(unique_actions))

    # Vectorize data
    for turn in high_fitness_data:
        # Features (X)
        state_features = flatten_state_for_nn(turn['state'], city_encoder)
        X_list.append(state_features[0]) # flatten_state_for_nn returns a 2D array
        
        # Labels (y)
        action_str = json.dumps(turn['action_taken'], sort_keys=True)
        y_list.append(action_encoder.transform([action_str])[0])

        # Weights
        weights.append(turn['final_fitness'])
        
    # Normalize weights
    min_weight = min(weights) if weights else 0
    normalized_weights = np.array(weights) - min_weight

    return np.array(X_list), np.array(y_list), normalized_weights, city_encoder, action_encoder


def train_next_generation(generation_num):
    """Loads data, trains a new model, and saves it for the next generation."""
    
    # 1. Load data from the specified generation
    data_path = f"data/generation_{generation_num}/simulation_data.json"
    sim_data = load_data(data_path)
    if not sim_data:
        return

    # 2. Preprocess the data
    all_cities = get_all_cities()
    X, y, weights, city_enc, action_enc = preprocess_training_data(sim_data, all_cities)
    
    if X.shape[0] == 0:
        print("No valid data available for training.")
        return

    # 3. Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Train the model
    model = MLPClassifier(random_state=42, max_iter=500, hidden_layer_sizes=(100, 100, 100, 100), alpha=0.001)

    # --- Diagnostic Prints ---
    print("\n--- Training Details ---")
    print(f"Training data shape: {X_train_scaled.shape}")
    print(f"Model parameters: {model.get_params()}")
    print("------------------------")
    
    print(f"Training model for Generation {generation_num + 1}...")
    model.fit(X_train_scaled, y_train) 

    # 5. Evaluate and report
    print("\nModel Evaluation:")
    y_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred, labels=action_enc.transform(action_enc.classes_), target_names=action_enc.classes_, zero_division=0))

    # 6. Save the new model and transformers
    output_dir = f"models/generation_{generation_num + 1}"
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, f"{output_dir}/pandemic_model.joblib")
    joblib.dump(scaler, f"{output_dir}/pandemic_model_scaler.joblib")
    joblib.dump(city_enc, f"{output_dir}/pandemic_model_city_encoder.joblib")
    joblib.dump(action_enc, f"{output_dir}/pandemic_model_action_encoder.joblib")
    
    print(f"New model saved for Generation {generation_num + 1} in '{output_dir}'")

if __name__ == "__main__":
    # Specify which generation's data you want to train on
    # This will create the model for GENERATION + 1
    GENERATION_TO_TRAIN_ON = 0
    train_next_generation(GENERATION_TO_TRAIN_ON)
