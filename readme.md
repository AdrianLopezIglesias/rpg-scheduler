# Pandemic AI Trainer

This project uses a generational learning approach to train a neural network to play a simplified version of the Pandemic board game efficiently.

### Project Structure

```
.
├── config.json
├── main.py                 # <-- Single entry point for all commands
├── requirements.txt
├── game/
│   └── ... (game files)
├── agents/
│   └── ... (agent files)
├── modules/
│   ├── __init__.py
│   ├── analysis.py         # <-- Contains logic for analyzing game results
│   ├── orchestrator.py     # <-- Contains the main training & calibration loops
│   ├── simulation_runner.py
│   ├── tester.py           # <-- Contains the `test` and `debug` logic
│   ├── trainer.py          # <-- Contains the model training logic
│   └── utils.py            # <-- Contains helper functions like log()
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

* **Train a new model from scratch:**
    ```bash
    python main.py train
    ```

* **Test a trained model's performance:**
    ```bash
    python main.py test
    ```

* **Debug a model's decisions:**
    ```bash
    python main.py debug
    ```

* **Calibrate to find the best network architecture:**
    ```bash
    python main.py calibrate
    