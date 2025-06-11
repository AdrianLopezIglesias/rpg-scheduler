import pandas as pd
import joblib
import random
import os

def get_path(filename):
    return os.path.join(os.path.dirname(__file__), filename)

def predict_action_dt(game_state):
    try:
        model = joblib.load(get_path('rpg_model_dt.joblib'))
        action_encoder = joblib.load(get_path('action_encoder_dt.joblib'))
        element_encoder = joblib.load(get_path('element_encoder_dt.joblib'))
    except FileNotFoundError:
        print("Model or encoder files not found.")
        print("Please run trainer_dt.py first to train and save the model.")
        return None

    flat_state = {
        'playerLevel': game_state['playerLevel'],
        'nextChapterLevel': game_state['nextChapterLevel'],
        'nextChapterElement': game_state['nextChapterElement'],
    }
    for element, level in game_state['playerElementLevel'].items():
        flat_state[f'element_{element}'] = level
    input_df = pd.DataFrame([flat_state])
    input_df['nextChapterElement_encoded'] = element_encoder.transform(input_df['nextChapterElement'])
    
    features = ['playerLevel', 'nextChapterLevel', 'nextChapterElement_encoded', 
                'element_fire', 'element_water', 'element_earth', 'element_wind']
    input_for_prediction = input_df[features]

    prediction_encoded = model.predict(input_for_prediction)
    prediction = action_encoder.inverse_transform(prediction_encoded)
    return prediction[0]

if __name__ == "__main__":
    elements = ["fire", "water", "earth", "wind"]
    player_level = random.randint(1, 20)
    new_game_state = {
        "playerLevel": player_level,
        "playerElementLevel": {el: random.randint(1, 10) for el in elements},
        "nextChapterLevel": player_level + random.randint(3, 10),
        "nextChapterElement": random.choice(elements)
    }

    predicted_action = predict_action_dt(new_game_state)
    
    if predicted_action:
        print("--- Sample Game State (Decision Tree) ---")
        print(new_game_state)
        print("\n--- Prediction ---")
        print(f"Predicted next action: {predicted_action}")
