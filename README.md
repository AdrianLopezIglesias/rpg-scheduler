# AI Task Scheduler for RPG

This project contains the necessary scripts to generate synthetic data and train a machine learning model to predict the optimal next action for a player in a fictional RPG.

## Project Structure

```
rpg-ai-scheduler/
├── requirements.txt       # Project dependencies
├── README.md              # This file
├── rpg_ai/
│   ├── __init__.py
│   ├── data_generator.py  # Script to generate synthetic_data.json
│   ├── trainer.py         # Script to train the model and save artifacts
│   └── predictor.py       # Script to load the model and predict an action
└── rpg_model.joblib       # Saved trained model (after running trainer.py)
└── action_encoder.joblib  # Saved action encoder (after running trainer.py)
└── element_encoder.joblib # Saved element encoder (after running trainer.py)
```

## Requirements

* Python 3.9 or higher.

## Setup and Installation

This project uses `venv` for environment management.

1.  **Navigate to the project directory:**
    ```bash
    cd rpg-ai-scheduler
    ```

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv .venv
    ```

3.  **Activate the virtual environment:**
    * On macOS/Linux (bash):
        ```bash
        source .venv/bin/activate
        ```
    * On macOS/Linux (fish):
        ```bash
        source .venv/bin/activate.fish
        ```
    * On Windows (Command Prompt/PowerShell):
        ```bash
        .\.venv\Scripts\activate
        ```

4.  **Install Dependencies:**
    With the virtual environment active, install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## How to Use

Make sure your virtual environment is active before running the scripts.

1.  **Generate Synthetic Data:**
    This script creates the `synthetic_data.json` file.
    ```bash
    python rpg_ai/data_generator.py
    ```

2.  **Train the Model:**
    This script loads the data, trains a model, and saves `rpg_model.joblib` and the encoder files.
    ```bash
    python rpg_ai/trainer.py
    ```

3.  **Test the Predictor:**
    This script loads the saved model and predicts an action for a randomly generated game state.
    ```bash
    python rpg_ai/predictor.py
    