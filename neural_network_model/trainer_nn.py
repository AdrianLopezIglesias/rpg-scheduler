# --- 1. Import Necessary Libraries ---
# These libraries provide the tools needed for data handling, machine learning, and file path management.
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier      # The Neural Network model.
from sklearn.preprocessing import StandardScaler, LabelEncoder # StandardScaler for data normalization, LabelEncoder for converting text to numbers.
from sklearn.metrics import classification_report
import joblib
import os # Used to handle file paths in a way that works on any operating system.

# --- 2. Define Helper and Data Loading Functions ---
def get_path(filename):
    """
    Creates a full file path by joining the script's directory with a filename.
    Why: This makes the script portable. It will find the files correctly whether you run it from the project root or from within its own folder.
    """
    return os.path.join(os.path.dirname(__file__), filename)

def load_data(filepath):
    """
    Loads the dataset from a specified JSON file.
    Why: We need to get our raw data from the file system into our script so we can work with it.
    """
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        print("Please run data_generator/data_generator.py first.")
        return None

# --- 3. Define Data Preprocessing Function ---
def preprocess_data(data):
    """
    Cleans and transforms the raw data into a format suitable for machine learning.
    Why: Machine learning models require structured, numerical data. This function flattens the nested JSON and converts text labels into numbers.
    """
    records = []
    for item in data:
        state = item['gameState']
        # Flatten the nested gameState dictionary into a single-level dictionary.
        flat_state = {
            'playerLevel': state['playerLevel'],
            'nextChapterLevel': state['nextChapterLevel'],
            'nextChapterElement': state['nextChapterElement']
        }
        for element, level in state['playerElementLevel'].items():
            flat_state[f'element_{element}'] = level
        flat_state['nextAction'] = item['nextAction']
        records.append(flat_state)
    
    df = pd.DataFrame(records) # Convert the list of flat dictionaries into a structured table (DataFrame).
    
    # Initialize encoders to map text categories to numbers.
    action_encoder = LabelEncoder()
    element_encoder = LabelEncoder()
    # Create new columns with the numerical representations of the text columns.
    df['nextAction_encoded'] = action_encoder.fit_transform(df['nextAction'])
    df['nextChapterElement_encoded'] = element_encoder.fit_transform(df['nextChapterElement'])
    return df, action_encoder, element_encoder

# --- 4. Define Model Training Function ---
def train_neural_network(df, action_encoder, element_encoder):
    """
    Trains the Neural Network, evaluates its performance, and saves the resulting model and transformers.
    Why: This is the core of the project where the model learns from the data.
    """
    if df is None or df.empty:
        print("DataFrame is empty. Cannot train model.")
        return

    # Define the features (inputs) and the target (output) for the model.
    features = ['playerLevel', 'nextChapterLevel', 'nextChapterElement_encoded', 
                'element_fire', 'element_water', 'element_earth', 'element_wind']
    X = df[features]  # The input data (game state).
    y = df['nextAction_encoded'] # The output data we want to predict (the next action).

    # Split data into a training set (for learning) and a testing set (for evaluation).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Feature Scaling ---
    # Why: Neural Networks perform best when input features are on a similar scale. 
    # StandardScaler transforms the data to have a mean of 0 and a standard deviation of 1.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) # Fit the scaler on training data and transform it.
    X_test_scaled = scaler.transform(X_test)       # Apply the same transformation to the test data.

    print("Training Neural Network (MLPClassifier)...")
    # Initialize the Neural Network model.
    # hidden_layer_sizes=(100, 50) defines the architecture: two hidden layers, the first with 100 neurons, the second with 50.
    # max_iter=1000 allows the model to go through the data up to 1000 times to find the best solution.
    # random_state=42 ensures the results are reproducible.
    model = MLPClassifier(random_state=42, max_iter=1000, hidden_layer_sizes=(1, 1))
    # Train the model on the scaled training data.
    model.fit(X_train_scaled, y_train)

    # --- Model Evaluation ---
    print("\nModel Evaluation:")
    y_pred = model.predict(X_test_scaled) # Make predictions on the unseen, scaled test data.
    # Convert numerical predictions back to text labels for a human-readable report.
    y_test_labels = action_encoder.inverse_transform(y_test)
    y_pred_labels = action_encoder.inverse_transform(y_pred)
    # Print the performance report.
    print(classification_report(y_test_labels, y_pred_labels, zero_division=0))
    
    # --- 5. Save the Artifacts ---
    # Why: We save the trained model and the transformers (encoders, scaler) so they can be loaded and used for predictions in another script without retraining.
    joblib.dump(model, get_path('rpg_model_nn.joblib'))
    joblib.dump(action_encoder, get_path('action_encoder_nn.joblib'))
    joblib.dump(element_encoder, get_path('element_encoder_nn.joblib'))
    joblib.dump(scaler, get_path('scaler_nn.joblib')) # The scaler must also be saved.
    print(f"\nNeural Network model, encoders, and scaler saved to: {os.path.dirname(__file__)}")

# --- 6. Main Execution Block ---
# This code runs only when the script is executed directly from the command line.
if __name__ == "__main__":
    # Define the path to the data file relative to this script.
    data_path = os.path.join(os.path.dirname(__file__), '..', 'synthetic_data.json')
    # Run the full pipeline: load data, preprocess it, and train the model.
    raw_data = load_data(data_path)
    if raw_data:
        processed_df, action_enc, elem_enc = preprocess_data(raw_data)
        train_neural_network(processed_df, action_enc, elem_enc)
