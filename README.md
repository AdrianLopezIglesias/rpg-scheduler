# AI Task Scheduler for RPG

This project trains and compares two different machine learning models (a Decision Tree and a Neural Network) to predict the optimal next action for a player in a fictional RPG.

## Project Structure

The project is now organized by functionality:

```
rpg-ai-scheduler/
├── requirements.txt           # Project dependencies
├── README.md                  # This file
├── data_generator/
│   └── data_generator.py      # Script to generate synthetic_data.json
├── decision_tree_model/
│   ├── trainer_dt.py          # Trains the Decision Tree model
│   └── predictor_dt.py        # Predicts actions using the DT model
└── neural_network_model/
    ├── trainer_nn.py          # Trains the Neural Network model
    └── predictor_nn.py        # Predicts actions using the NN model
```
The trained model files (e.g., `rpg_model_dt.joblib`) will be saved inside their respective model folders.

## Requirements

* Python 3.9 or higher.

## Setup and Installation

1.  **Navigate to the project directory:**
    ```bash
    cd rpg-ai-scheduler
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    # Create
    python3 -m venv .venv
    # Activate (macOS/Linux)
    source .venv/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Use

Make sure your virtual environment is active.

1.  **Generate Synthetic Data:**
    Run this script first. It will create `synthetic_data.json` in the root directory.
    ```bash
    python data_generator/data_generator.py
    ```
2.  **Train a Model:**
    Choose which model you want to train.
    * **Decision Tree:**
        ```bash
        python decision_tree_model/trainer_dt.py
        ```
    * **Neural Network:**
        ```bash
        python neural_network_model/trainer_nn.py
        ```
3.  **Test a Predictor:**
    After training a model, you can test its corresponding predictor.
    * **Decision Tree:**
        ```bash
        python decision_tree_model/predictor_dt.py
        ```
    * **Neural Network:**
        ```bash
        python neural_network_model/predictor_nn.py
        