import os
import json
import numpy as np
import time
from datetime import datetime
from simulation_runner import run_simulation
import trainer
from agents.agents import RandomAgent, NNAgent

def log(message):
    """Prints a message with a timestamp."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def analyze_generation_data(filepath):
    """Analyzes the game data format for key performance metrics."""
    try:
        list_of_games = json.load(open(filepath, 'r'))
    except (FileNotFoundError, json.JSONDecodeError):
        return

    num_games = len(list_of_games)
    if num_games == 0:
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
    
    print("\n--- Generation Analysis ---")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Average Actions to Win: {avg_win_speed}")
    print(f"Fastest Win: {fastest_win} actions")
    print("---------------------------\n")

def main_loop(curriculum, games_per_generation):
    """Runs the main generational training loop using a curriculum of difficulties."""
    total_generations = len(curriculum)
    epsilon_start = 0.5
    epsilon_end = 0.05
    epsilon_decay = (epsilon_start - epsilon_end) / total_generations if total_generations > 0 else 0

    for gen, difficulty in enumerate(curriculum):
        log(f"STARTING GENERATION {gen} on '{difficulty}' map")
        
        sim_output_path = f"data/{difficulty}/generation_{gen}/simulation_data.json"
        
        start_time = time.time()

        if gen == 0:
            agent = RandomAgent()
        else:
            # Always load the model from the previous generation's difficulty
            prev_difficulty = curriculum[gen - 1]
            model_path_prefix = f"models/{prev_difficulty}/generation_{gen}/pandemic_model"
            current_epsilon = epsilon_start - (gen * epsilon_decay)
            agent = NNAgent(model_path_prefix, epsilon=current_epsilon)
            if not agent.model:
                log(f"Failed to load model for Gen {gen}. Trying RandomAgent.")
                agent = RandomAgent()
        
        run_simulation(agent, games_per_generation, sim_output_path, difficulty=difficulty)
        sim_duration = time.time() - start_time
        log(f"Simulation for Gen {gen} finished in {sim_duration:.2f} seconds.")

        analyze_generation_data(sim_output_path)
        
        start_time = time.time()
        trainer.train_next_generation(gen, difficulty=difficulty)
        train_duration = time.time() - start_time
        log(f"Training for Gen {gen + 1} finished in {train_duration:.2f} seconds.")
        log(f"FINISHED GENERATION {gen}\n")

if __name__ == "__main__":
    # Define the training curriculum. The agent will train on these maps in sequence.
    # We train on 'easy' for several generations, then introduce 'hard'.
    TRAINING_CURRICULUM = [
        "easy", "easy", "easy", "easy", "easy", # 5 generations on the easy map
        "hard", "hard", "hard", "hard", "hard"  # 5 generations on the hard map
    ]
    GAMES_PER_GENERATION = 1000

    main_loop(TRAINING_CURRICULUM, GAMES_PER_GENERATION)
