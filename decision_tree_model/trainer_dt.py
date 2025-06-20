import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# --- Helper function to get correct file paths ---
def get_path(filename):
    # Joins the directory of this script with the filename
    return os.path.join(os.path.dirname(__file__), filename)

def load_data(filepath):
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        print("Please run data_generator/data_generator.py first to create the dataset.")
        return None

def preprocess_data(data):
    records = []
    for item in data:
        state = item['gameState']
        flat_state = {
            'playerLevel': state['playerLevel'],
            'nextChapterLevel': state['nextChapterLevel'],
            'nextChapterElement': state['nextChapterElement']
        }
        for element, level in state['playerElementLevel'].items():
            flat_state[f'element_{element}'] = level
        flat_state['nextAction'] = item['nextAction']
        records.append(flat_state)
    df = pd.DataFrame(records)
    action_encoder = LabelEncoder()
    element_encoder = LabelEncoder()
    df['nextAction_encoded'] = action_encoder.fit_transform(df['nextAction'])
    df['nextChapterElement_encoded'] = element_encoder.fit_transform(df['nextChapterElement'])
    return df, action_encoder, element_encoder

def train_decision_tree(df, action_encoder, element_encoder):
    if df is None or df.empty:
        print("DataFrame is empty. Cannot train model.")
        return

    features = ['playerLevel', 'nextChapterLevel', 'nextChapterElement_encoded', 
                'element_fire', 'element_water', 'element_earth', 'element_wind']
    X = df[features]
    y = df['nextAction_encoded']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Decision Tree model...")
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    print("\nModel Evaluation:")
    y_pred = model.predict(X_test)
    y_test_labels = action_encoder.inverse_transform(y_test)
    y_pred_labels = action_encoder.inverse_transform(y_pred)
    print(classification_report(y_test_labels, y_pred_labels, zero_division=0))
    
    joblib.dump(model, get_path('rpg_model_dt.joblib'))
    joblib.dump(action_encoder, get_path('action_encoder_dt.joblib'))
    joblib.dump(element_encoder, get_path('element_encoder_dt.joblib'))
    print(f"\nDecision Tree model and encoders saved to: {os.path.dirname(__file__)}")

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), '..', 'synthetic_data.json')
    raw_data = load_data(data_path)
    if raw_data:
        processed_df, action_enc, elem_enc = preprocess_data(raw_data)
        train_decision_tree(processed_df, action_enc, elem_enc)
