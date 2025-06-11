import pandas as pd
import joblib
import random
import os

# --- World Constants - Must match data_generator ---
LOCATIONS = ["City", "Volcano", "Mines", "Forest", "Mountains"]
ELEMENTS = ["fire", "water", "earth", "wind"]

def get_path(filename):
    return os.path.join(os.path.dirname(__file__), filename)

def predict_action_nn(game_state):
    try:
        # Load all the saved artifacts
        model = joblib.load(get_path('rpg_model_nn.joblib'))
        action_encoder = joblib.load(get_path('action_encoder_nn.joblib'))
        element_encoder = joblib.load(get_path('element_encoder_nn.joblib'))
        location_encoder = joblib.load(get_path('location_encoder_nn.joblib')) # Load location encoder
        scaler = joblib.load(get_path('scaler_nn.joblib'))
    except FileNotFoundError:
        print("Model, encoder, or scaler files not found.")
        print("Please run trainer_nn.py first to train and save the artifacts.")
        return None

    # Flatten the game state into a dictionary
    flat_state = {
        'playerLevel': game_state['playerLevel'],
        'nextChapterLevel': game_state['nextChapterLevel'],
        'nextChapterElement': game_state['nextChapterElement'],
        'playerLocation': game_state['playerLocation'],
        'nextChapterLocation': game_state['nextChapterLocation']
    }
    for element, level in game_state['playerElementLevel'].items():
        flat_state[f'element_{element}'] = level
    
    input_df = pd.DataFrame([flat_state])
    
    # Encode all categorical features using the loaded encoders
    input_df['nextChapterElement_encoded'] = element_encoder.transform(input_df['nextChapterElement'])
    input_df['playerLocation_encoded'] = location_encoder.transform(input_df['playerLocation'])
    input_df['nextChapterLocation_encoded'] = location_encoder.transform(input_df['nextChapterLocation'])
    
    # Define the full feature set to match the training script
    features = [
        'playerLevel', 'nextChapterLevel', 'nextChapterElement_encoded', 
        'element_fire', 'element_water', 'element_earth', 'element_wind',
        'playerLocation_encoded', 'nextChapterLocation_encoded'
    ]
    
    # Ensure the columns are in the correct order
    input_df_features = input_df[features]

    # Scale the input data using the loaded scaler
    input_df_scaled = scaler.transform(input_df_features)

    # Make the prediction
    prediction_encoded = model.predict(input_df_scaled)
    prediction = action_encoder.inverse_transform(prediction_encoded)
    return prediction[0]

if __name__ == "__main__":
    # Generate a random game state consistent with the new complex world
    player_level = random.randint(1, 20)
    new_game_state = {
        "playerLevel": player_level,
        "playerElementLevel": {el: random.randint(1, 20) for el in ELEMENTS},
        "nextChapterLevel": random.uniform(player_level + 5, player_level + 25),
        "nextChapterElement": random.choice(ELEMENTS),
        "playerLocation": random.choice(LOCATIONS),
        "nextChapterLocation": random.choice(LOCATIONS)
    }

    predicted_action = predict_action_nn(new_game_state)
    
    if predicted_action:
        print("--- Sample Game State (Complex Neural Network) ---")
        for key, value in new_game_state.items():
            print(f"  {key}: {value}")
        print("\n--- Prediction ---")
        print(f"Predicted next action: {predicted_action}")
