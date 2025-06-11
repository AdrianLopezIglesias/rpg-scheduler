# --- 1. Import Necessary Libraries ---
# These libraries provide the tools needed for data manipulation, machine learning, and saving our model.
import json  # To read and write data in JSON format.
import pandas as pd  # A powerful library for data analysis and manipulation, used here to structure our data in a table (DataFrame).
from sklearn.model_selection import train_test_split  # A function to split our data into training and testing sets.
from sklearn.tree import DecisionTreeClassifier  # The machine learning algorithm we'll use to make predictions.
from sklearn.metrics import classification_report  # A tool to evaluate how well our model performed.
from sklearn.preprocessing import LabelEncoder  # A utility to convert text labels (like "train") into numbers that the model can understand.
import joblib  # A set of tools to efficiently save and load Python objects, perfect for saving our trained model.

# --- 2. Define Data Loading Function ---
def load_data(filepath="synthetic_data.json"):
    """
    Loads the dataset from a specified JSON file.
    Why: We need to get our raw data from the file system into our script so we can work with it.
    """
    try:
        # Open the file in read mode ("r") and use json.load to parse it.
        with open(filepath, "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        # If the file doesn't exist, we can't proceed. This provides a clear error message to the user.
        print(f"Error: {filepath} not found.")
        print("Please run data_generator.py first to create the dataset.")
        return None

# --- 3. Define Data Preprocessing Function ---
def preprocess_data(data):
    """
    Cleans and transforms the raw data into a format suitable for machine learning.
    Why: Raw data is often not in the right shape for a model. This function flattens the nested data and converts text to numbers.
    """
    records = []
    # Loop through each sample in our raw data.
    for item in data:
        state = item['gameState']
        # Create a "flat" dictionary. Machine learning models work best with tabular data (rows and columns), not nested structures.
        flat_state = {
            'playerLevel': state['playerLevel'],
            'nextChapterLevel': state['nextChapterLevel'],
            'nextChapterElement': state['nextChapterElement']
        }
        # Un-nest the element levels, turning each element into its own column (e.g., 'element_fire').
        for element, level in state['playerElementLevel'].items():
            flat_state[f'element_{element}'] = level
        
        # Add the target variable (what we want to predict) to our flat dictionary.
        flat_state['nextAction'] = item['nextAction']
        records.append(flat_state)
    
    # Convert the list of dictionaries into a pandas DataFrame. This is like a spreadsheet in Python.
    df = pd.DataFrame(records)
    
    # Initialize encoders. These will create the mapping from text to numbers.
    action_encoder = LabelEncoder()
    element_encoder = LabelEncoder()

    # Create new columns with the encoded (numerical) versions of our text columns.
    # The fit_transform method learns the mapping and applies it in one step.
    df['nextAction_encoded'] = action_encoder.fit_transform(df['nextAction'])
    df['nextChapterElement_encoded'] = element_encoder.fit_transform(df['nextChapterElement'])

    # Return the processed DataFrame and the encoders so we can use them later (especially for decoding predictions).
    return df, action_encoder, element_encoder

# --- 4. Define Model Training Function ---
def train_model(df, action_encoder, element_encoder):
    """
    Trains the machine learning model, evaluates its performance, and saves the results.
    Why: This is the core of our project. It's where the machine "learns" from the data.
    """
    if df is None or df.empty:
        print("DataFrame is empty. Cannot train model.")
        return

    # Define which columns are our "features" (inputs for the model).
    features = ['playerLevel', 'nextChapterLevel', 'nextChapterElement_encoded', 
                'element_fire', 'element_water', 'element_earth', 'element_wind']
    
    # X contains our input features.
    X = df[features]
    # y contains our target variable (the thing we want to predict).
    y = df['nextAction_encoded']
    
    # Split the data. 80% goes to training, 20% to testing.
    # Why: We need to test the model on data it has never seen before to get a true sense of its performance.
    # random_state=42 ensures the split is the same every time we run the script, making our results reproducible.
    # stratify=y ensures the proportion of each action is the same in both the training and testing sets, which is important for imbalanced data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Decision Tree model...")
    # Initialize the Decision Tree model.
    model = DecisionTreeClassifier(random_state=42)
    # "Fit" the model to the training data. This is the learning process.
    model.fit(X_train, y_train)

    print("\nModel Evaluation:")
    # Ask the trained model to make predictions on the unseen test data.
    y_pred = model.predict(X_test)
    
    # To make the report human-readable, we convert the numerical predictions back to their original text labels.
    y_test_labels = action_encoder.inverse_transform(y_test)
    y_pred_labels = action_encoder.inverse_transform(y_pred)

    # Print a detailed report of the model's accuracy, precision, and other key metrics.
    # zero_division=0 prevents a warning if a class has no predictions, which can happen in some cases.
    print(classification_report(y_test_labels, y_pred_labels, zero_division=0))
    
    # --- 5. Save the Trained Model and Encoders ---
    # Why: Saving these "artifacts" allows us to use them in another script (predictor.py) without having to retrain the model every time.
    joblib.dump(model, 'rpg_model.joblib')
    joblib.dump(action_encoder, 'action_encoder.joblib')
    joblib.dump(element_encoder, 'element_encoder.joblib')
    print("\nModel and encoders saved to disk.")

# --- 6. Main Execution Block ---
# This ensures the code inside only runs when the script is executed directly (not when imported as a module).
if __name__ == "__main__":
    # Orchestrate the entire process: load, preprocess, and train.
    print("Starting model training process...")
    raw_data = load_data()
    if raw_data: # Only proceed if data was loaded successfully.
        processed_df, action_enc, elem_enc = preprocess_data(raw_data)
        train_model(processed_df, action_enc, elem_enc)
    print("Process finished.")
