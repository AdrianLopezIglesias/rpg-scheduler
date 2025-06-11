import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import joblib
import os
import numpy as np

def load_data(filepath="simulation_data_fitness.json"):
    """Loads the simulation data from the specified JSON file."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Please run the simulator first.")
        return None

def flatten_state(state, city_encoder):
    """Converts the nested state dictionary into a flat numerical vector."""
    board_state = state['board']
    # Create a feature for each city's cube count
    cube_features = [board_state.get(city, {}).get('cubes', 0) for city in city_encoder.classes_]
    
    # Encode the player's location
    player_location_encoded = city_encoder.transform([state['player_location']])[0]
    
    # Combine all features into a single list
    features = cube_features + [player_location_encoded, state['current_round']]
    return features

def preprocess_data(data, all_cities):
    """
    Processes the raw simulation data into features (X), labels (y),
    and sample weights based on fitness scores.
    """
    print("Preprocessing data...")
    X_list = []
    y_list = []
    weights_list = []

    # Create encoders for categorical data
    city_encoder = LabelEncoder().fit(all_cities)
    
    # To encode actions, we need to find all possible unique actions first
    unique_actions = set()
    for turn in data:
        # We focus on the first action as the primary decision for this turn
        action = turn['playerActions'][0]
        unique_actions.add(json.dumps(action, sort_keys=True))
    
    action_encoder = LabelEncoder().fit(list(unique_actions))

    for turn in data:
        # Create features (X)
        state_features = flatten_state(turn['initialState'], city_encoder)
        X_list.append(state_features)
        
        # Create labels (y) - predicting the first action of the sequence
        action = json.dumps(turn['playerActions'][0], sort_keys=True)
        y_list.append(action_encoder.transform([action])[0])
        
        # Use the fitness score as a weight for this training sample
        weights_list.append(turn['fitnessScore'])

    # Normalize weights to prevent extreme values, ensuring they are non-negative
    if not weights_list:
        return np.array([]), np.array([]), np.array([]), city_encoder, action_encoder

    min_weight = min(weights_list) if weights_list else 0
    weights_np = np.array(weights_list) - min_weight
    
    return np.array(X_list), np.array(y_list), weights_np, city_encoder, action_encoder


def train_pandemic_nn(X, y, weights, action_encoder, city_encoder):
    """
    Trains, evaluates, and saves the Neural Network model for Pandemic.
    """
    print("Splitting data and scaling features...")
    # We no longer need w_test, so we can use '_' to ignore it.
    X_train, X_test, y_train, y_test, w_train, _ = train_test_split(
        X, y, weights, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --- Resampling based on weights ---
    # Since MLPClassifier.fit() does not support 'sample_weight', we resample the training data.
    # This creates a new training set where samples with higher fitness scores are more likely to appear.
    print("Resampling training data based on fitness scores...")
    total_weight = np.sum(w_train)
    if total_weight > 0:
        probabilities = w_train / total_weight
        resampled_indices = np.random.choice(len(X_train_scaled), size=len(X_train_scaled), p=probabilities, replace=True)
        X_train_resampled = X_train_scaled[resampled_indices]
        y_train_resampled = y_train[resampled_indices]
    else:
        # If all weights are zero, use the original training set
        X_train_resampled = X_train_scaled
        y_train_resampled = y_train

    print("Training Neural Network (MLPClassifier)...")
    model = MLPClassifier(random_state=42, max_iter=1000, hidden_layer_sizes=(50, 25), alpha=0.001)
    
    # Train on the resampled data, without the sample_weight argument
    model.fit(X_train_resampled, y_train_resampled)
    
    print("\nModel Evaluation:")
    y_pred = model.predict(X_test_scaled)

    # Decode the numerical labels back to their original string representation
    y_test_labels = action_encoder.inverse_transform(y_test)
    y_pred_labels = action_encoder.inverse_transform(y_pred)
    
    # Get all possible labels for a complete report
    all_labels = action_encoder.classes_

    print(classification_report(y_test_labels, y_pred_labels, labels=all_labels, zero_division=0))

    # Save all necessary artifacts for prediction
    output_dir = "pandemic_model"
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_dir, 'pandemic_model.joblib'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
    joblib.dump(city_encoder, os.path.join(output_dir, 'city_encoder.joblib'))
    joblib.dump(action_encoder, os.path.join(output_dir, 'action_encoder.joblib'))
    
    print(f"Model and transformers saved to '{output_dir}' directory.")


if __name__ == "__main__":
    simulation_data = load_data()
    
    if simulation_data:
        # Define all cities from the map to ensure consistent encoding
        # This should ideally be loaded from a shared config, but is hardcoded for simplicity.
        all_cities_list = [
            "San Francisco", "Chicago", "Atlanta", "Montreal", "Washington", 
            "New York", "Los Angeles", "Mexico City", "Miami", "Tokyo", 
            "Manila", "London", "Madrid", "Sydney", "Lima", "Bogota"
        ]
        
        X_data, y_data, sample_weights, city_encoder, action_encoder = preprocess_data(simulation_data, all_cities_list)
        
        if X_data.size > 0:
            # Pass the fitted encoders to the training function so they can be used and saved
            train_pandemic_nn(X_data, y_data, sample_weights, action_encoder, city_encoder)
        else:
            print("No data available to train the model.")
