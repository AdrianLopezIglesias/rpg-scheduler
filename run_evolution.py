import os
import json
import numpy as np
from simulation_runner import run_simulation
from trainer import train_next_generation
from agents.agents import RandomAgent, NNAgent

def analyze_generation_data(filepath):
    """Analyzes the new game data format for key performance metrics."""
    try:
        list_of_games = json.load(open(filepath, 'r'))
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Analysis failed: Could not read data from {filepath}")
        return

    num_games = len(list_of_games)
    if num_games == 0:
        return

    winning_games = [g for g in list_of_games if g.get("final_result") == "win"]
    win_rate = len(winning_games) / num_games * 100
    
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

def main_loop(total_generations, games_per_generation):
    """Runs the main generational training loop with the new logic."""
    for gen in range(total_generations):
        print(f"=============== STARTING GENERATION {gen} ===============")

        if gen == 0:
            agent = RandomAgent()
        else:
            model_path_prefix = f"models/generation_{gen}/pandemic_model"
            agent = NNAgent(model_path_prefix)
            if not agent.model:
                print(f"Failed to load model for Gen {gen}. Halting.")
                break
        
        sim_output_path = f"data/generation_{gen}/simulation_data.json"
        run_simulation(agent, games_per_generation, sim_output_path)
        analyze_generation_data(sim_output_path)
        train_next_generation(gen)
        print(f"=============== FINISHED GENERATION {gen} ===============\n")

if __name__ == "__main__":
    TOTAL_GENERATIONS = 10
    GAMES_PER_GENERATION = 500

    main_loop(TOTAL_GENERATIONS, GAMES_PER_GENERATION)
