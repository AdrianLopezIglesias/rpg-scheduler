# Pandemic AI Trainer

This project uses a generational learning approach to train a neural network to play a simplified version of the Pandemic board game efficiently.

### Project Structure

The project has been refactored into a clean, modular structure:

```
.
├── config.json             # <-- All tunable parameters here
├── main.py                 # <-- The new single entry point for all actions
├── requirements.txt
├── game/
│   ├── __init__.py
│   ├── maps.json           # Map configurations
│   └── pandemic_game.py    # Core game logic
├── agents/
│   ├── __init__.py
│   └── agents.py           # Agent logic (Random, NN)
├── modules/
│   ├── __init__.py
│   ├── simulation_runner.py # Function to run simulations
│   └── trainer.py          # Function to train models
├── data/
│   └── ... (simulation data folders)
└── models/
    └── ... (trained model folders)
```

### How to Use

All operations are now handled by `main.py`.

**1. Configure Your Run**

Edit `config.json` to set parameters for your desired action.

**2. Execute an Action**

Choose one of the three commands:

* **Train a new model from scratch:**
    This runs the full generational evolution loop.
    ```bash
    python main.py train
    ```

* **Test a trained model's performance:**
    This runs a batch of games with a specific model and reports the statistics.
    ```bash
    python main.py test
    ```

* **Debug a model's decisions:**
    This runs a single game and prints the agent's thought process for each turn.
    ```bash
    python main.py debug
    