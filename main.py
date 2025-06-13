import json
import argparse
from modules.orchestrator import run_training_loop, run_calibration_loop
from modules.tester import run_test, run_debug
from modules.utils import log
# --- NEW IMPORT FOR PHASE 4 ---
from modules.rl_trainer import run_rl_training
# ------------------------------

def main():
    """
    Main entry point for the Pandemic AI CLI.
    """
    parser = argparse.ArgumentParser(description="Pandemic AI CLI")
    parser.add_argument("command", choices=["train", "test", "debug", "calibrate", "train_rl"], help="The action to perform.")
    args = parser.parse_args()

    try:
        with open("config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        log("Error: config.json not found. Please create it.")
        return

    # --- MODIFIED FOR PHASE 4 ---
    # Added 'train_rl' command to launch the new trainer.
    if args.command == "train_rl":
        run_rl_training(config)
    # ----------------------------
    elif args.command == "train":
        log("Running original supervised training loop.")
        run_training_loop(config)
    elif args.command == "calibrate":
        run_calibration_loop(config)
    elif args.command == "test":
        run_test(config)
    elif args.command == "debug":
        run_debug(config)

if __name__ == "__main__":
    main()