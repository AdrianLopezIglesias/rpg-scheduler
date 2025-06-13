import os
import json
import numpy as np
import time
from datetime import datetime
from simulation_runner import run_simulation
from agents.agents import NNAgent

def log(message):
    """Prints a message with a timestamp."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def analyze_test_data(filepath, model_gen, train_difficulty, test_difficulty):
    """Analyzes and prints the final performance of the agent."""
    try:
        list_of_games = json.load(open(filepath, 'r'))
    except (FileNotFoundError, json.JSONDecodeError):
        log(f"Analysis failed: Could not read data from {filepath}")
        return

    num_games = len(list_of_games)
    if num_games == 0:
        log("Analysis failed: No game data to analyze.")
        return

    winning_games = [g for g in list_of_games if g.get("final_result") == "win"]
    win_rate = len(winning_games) / num_games * 100 if num_games > 0 else 0
    
    if winning_games:
        win_lengths = [g["total_actions"] for g in winning_games]
        avg_win_speed = np.mean(win_lengths)
        fastest_win = min(win_lengths)
    else:
        avg_win_speed = "N/A"
        fastest_win = "N/A"
    
    print("\n--- Test Results ---")
    print(f"Model from Gen {model_gen} ('{train_difficulty}') tested on '{test_difficulty}' map:")
    print(f"  Win Rate: {win_rate:.2f}%")
    print(f"  Average Actions to Win: {avg_win_speed}")
    print(f"  Fastest Win: {fastest_win} actions")
    print("--------------------\n")

def test_model(model_gen, train_difficulty, test_difficulty, num_games):
    """
    Loads a specific trained model and tests its performance on a specified map.
    """
    log(f"Testing model from Gen {model_gen} ('{train_difficulty}') on '{test_difficulty}' map...")

    # Load the agent with the specified trained model
    model_path_prefix = f"models/{train_difficulty}/generation_{model_gen}/pandemic_model"
    agent_to_test = NNAgent(model_path_prefix, epsilon=0) # Epsilon=0 for pure exploitation

    if not agent_to_test.model:
        log(f"Could not load the specified model (Gen {model_gen}, Difficulty '{train_difficulty}'). Halting test.")
        return

    # Define where to save the test run data
    test_output_path = f"data/test_runs/gen_{model_gen}_{train_difficulty}_on_{test_difficulty}.json"
    
    # Run the simulation
    start_time = time.time()
    run_simulation(agent_to_test, num_games, test_output_path, difficulty=test_difficulty)
    duration = time.time() - start_time
    log(f"Test simulation finished in {duration:.2f} seconds.")

    # Analyze the results
    analyze_test_data(test_output_path, model_gen, train_difficulty, test_difficulty)


if __name__ == "__main__":
    # --- Configuration for Testing ---
    
    # 1. Specify the generation of the trained model you want to test
    MODEL_GENERATION_TO_TEST = 9 # Example: test the model from the 9th generation
    
    # 2. Specify the difficulty the model was TRAINED on
    MODEL_TRAINING_DIFFICULTY = "easy"
    
    # 3. Specify the difficulty you want to TEST the model on
    DIFFICULTY_TO_TEST_ON = "hard"
    
    # 4. Number of games to run for the test
    GAMES_TO_TEST = 1

    test_model(
        MODEL_GENERATION_TO_TEST,
        MODEL_TRAINING_DIFFICULTY,
        DIFFICULTY_TO_TEST_ON,
        GAMES_TO_TEST
    )
