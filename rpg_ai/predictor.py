import pandas as pd
import joblib
import random

def predict_action(game_state):
    """
    Predicts the next action for a given game state using the trained model.
    """
    try:
        # Load the trained model and encoders from disk
        model = joblib.load('rpg_model.joblib')
        action_encoder = joblib.load('action_encoder.joblib')
        element_encoder = joblib.load('element_encoder.joblib')
    except FileNotFoundError:
        print("Model or encoder files not found.")
        print("Please run trainer.py first to train and save the model.")
        return None

    # Prepare the input game state into a flat structure for the DataFrame
    flat_state = {
        'playerLevel': game_state['playerLevel'],
        'nextChapterLevel': game_state['nextChapterLevel'],
        'nextChapterElement': game_state['nextChapterElement'], # This was the missing key
    }
    
    # Add the player's element levels to the flat structure
    for element, level in game_state['playerElementLevel'].items():
        flat_state[f'element_{element}'] = level

    # Create a pandas DataFrame from the flattened state
    input_df = pd.DataFrame([flat_state])

    # Transform the categorical 'nextChapterElement' using the loaded encoder
    # The original error occurred because 'nextChapterElement' was not in the DataFrame
    input_df['nextChapterElement_encoded'] = element_encoder.transform(input_df['nextChapterElement'])
    
    # Define the feature list to ensure the column order matches the training data
    features = ['playerLevel', 'nextChapterLevel', 'nextChapterElement_encoded', 
                'element_fire', 'element_water', 'element_earth', 'element_wind']
    
    # Reorder the DataFrame columns to match the model's expected input
    input_for_prediction = input_df[features]

    # Use the model to predict the encoded action
    prediction_encoded = model.predict(input_for_prediction)
    
    # Decode the numeric prediction back to the original string label
    prediction = action_encoder.inverse_transform(prediction_encoded)
    
    return prediction[0]

if __name__ == "__main__":
    # Generate quasi-random stats for prediction
    elements = ["fire", "water", "earth", "wind"]
    player_level = random.randint(1, 20)

    new_game_state = {
        "playerLevel": player_level,
        "playerElementLevel": {el: random.randint(1, 10) for el in elements},
        "nextChapterLevel": player_level + random.randint(3, 10),
        "nextChapterElement": random.choice(elements)
    }

    # Get the predicted action
    predicted_action = predict_action(new_game_state)
    
    # Print the results
    if predicted_action:
        print("--- Sample Game State ---")
        print(new_game_state)
        print("\n--- Prediction ---")
        print(f"Predicted next action: {predicted_action}")
