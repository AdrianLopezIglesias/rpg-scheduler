import json
import argparse
from modules.orchestrator import run_training_loop, run_calibration_loop
from modules.tester import run_test, run_debug
from modules.playback import run_gnn_playback
from modules.utils import log
from modules.rl_trainer import run_rl_training
from modules.debug_trainer import run_single_train_step_debug
from modules.validator import run_validation
from modules.curriculum_trainer import run_curriculum_training

def main():
    parser = argparse.ArgumentParser(description="Pandemic AI CLI")
    parser.add_argument("command", choices=["train", "test", "debug", "calibrate", "train_rl", "playback", "debug_train_step", "validate", "train_curriculum"], help="The action to perform.")
    args = parser.parse_args()
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        log("Error: config.json not found. Please create it.")
        return

    if args.command == "train_curriculum":
        run_curriculum_training(config)
    elif args.command == "train_rl":
        run_rl_training(config)
    elif args.command == "playback":
        run_gnn_playback(config)
    elif args.command == "debug_train_step":
        run_single_train_step_debug(config)
    elif args.command == "validate":
        run_validation(config)
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