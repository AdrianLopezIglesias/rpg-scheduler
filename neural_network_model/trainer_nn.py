import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import joblib
import os

def get_path(filename):
    return os.path.join(os.path.dirname(__file__), filename)

def load_data(filepath):
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        print("Please run data_generator/data_generator.py first.")
        return None

def preprocess_data(data):
    """
    Updated to handle new location features from the complex data generator.
    """
    records = []
    for item in data:
        state = item['gameState']
        # Flatten all relevant features from the gameState
        flat_state = {
            'playerLevel': state['playerLevel'],
            'nextChapterLevel': state['nextChapterLevel'],
            'nextChapterElement': state['nextChapterElement'],
            'playerLocation': state['playerLocation'],
            'nextChapterLocation': state['nextChapterLocation'],
        }
        for element, level in state['playerElementLevel'].items():
            flat_state[f'element_{element}'] = level
        flat_state['nextAction'] = item['nextAction']
        records.append(flat_state)
    
    df = pd.DataFrame(records)
    
    # Initialize encoders for all categorical features
    action_encoder = LabelEncoder()
    element_encoder = LabelEncoder()
    location_encoder = LabelEncoder() # For both player and chapter locations

    # Fit and transform all categorical columns
    df['nextAction_encoded'] = action_encoder.fit_transform(df['nextAction'])
    df['nextChapterElement_encoded'] = element_encoder.fit_transform(df['nextChapterElement'])
    
    # Use the same encoder for both location columns to maintain consistent mapping
    all_locations = pd.concat([df['playerLocation'], df['nextChapterLocation']]).unique()
    location_encoder.fit(all_locations)
    df['playerLocation_encoded'] = location_encoder.transform(df['playerLocation'])
    df['nextChapterLocation_encoded'] = location_encoder.transform(df['nextChapterLocation'])

    return df, action_encoder, element_encoder, location_encoder

def train_neural_network(df, action_encoder, element_encoder, location_encoder):
    if df is None or df.empty:
        print("DataFrame is empty. Cannot train model.")
        return

    # Update feature list to include new encoded location features
    features = [
        'playerLevel', 'nextChapterLevel', 'nextChapterElement_encoded', 
        'element_fire', 'element_water', 'element_earth', 'element_wind',
        'playerLocation_encoded', 'nextChapterLocation_encoded'
    ]
    X = df[features]
    y = df['nextAction_encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training Neural Network (MLPClassifier) on complex data...")
    # Using a slightly larger network to handle the increased complexity
    model = MLPClassifier(random_state=42, max_iter=1000, hidden_layer_sizes=(15))
    model.fit(X_train_scaled, y_train)

    print("\nModel Evaluation:")
    y_pred = model.predict(X_test_scaled)
    y_test_labels = action_encoder.inverse_transform(y_test)
    y_pred_labels = action_encoder.inverse_transform(y_pred)
    # Ensure all possible labels are included for a complete report
    all_labels = action_encoder.classes_
    print(classification_report(y_test_labels, y_pred_labels, labels=all_labels, zero_division=0))
    
    # Save all necessary artifacts
    joblib.dump(model, get_path('rpg_model_nn.joblib'))
    joblib.dump(action_encoder, get_path('action_encoder_nn.joblib'))
    joblib.dump(element_encoder, get_path('element_encoder_nn.joblib'))
    joblib.dump(location_encoder, get_path('location_encoder_nn.joblib')) # Save the new location encoder
    joblib.dump(scaler, get_path('scaler_nn.joblib'))
    print(f"\nNeural Network model and all transformers saved to: {os.path.dirname(__file__)}")

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), '..', 'synthetic_data.json')
    raw_data = load_data(data_path)
    if raw_data:
        processed_df, action_enc, elem_enc, loc_enc = preprocess_data(raw_data)
        train_neural_network(processed_df, action_enc, elem_enc, loc_enc)
