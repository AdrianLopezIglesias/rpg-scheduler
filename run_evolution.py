import os
import json
import numpy as np
from simulation_runner import run_simulation
from trainer import train_next_generation
from agents.agents import RandomAgent, NNAgent

def analyze_generation_data(filepath):
    """
    Analyzes a generation's data file to extract and print key performance metrics.
    """
    try:
        with open(filepath, 'r') as f:
            list_of_games = json.load(f)
    except FileNotFoundError:
        print(f"Analysis failed: Data file not found at {filepath}")
        return

    if not list_of_games:
        print("Analysis failed: No game data to analyze.")
        return

    num_games = len(list_of_games)
    wins = 0
    fitness_scores = []

    for game in list_of_games:
        if not game:
            continue
        # The result and fitness are the same for every step in a game's history
        final_result = game[-1].get("final_result")
        final_fitness = game[-1].get("final_fitness")

        if final_result == 'win':
            wins += 1
        if final_fitness is not None:
            fitness_scores.append(final_fitness)

    win_rate = (wins / num_games) * 100 if num_games > 0 else 0
    avg_fitness = np.mean(fitness_scores) if fitness_scores else 0
    max_fitness = max(fitness_scores) if fitness_scores else 0

    print("\n--- Generation Analysis ---")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Average Fitness Score: {avg_fitness:.2f}")
    print(f"Maximum Fitness Score: {max_fitness:.2f}")
    print("---------------------------\n")


def main_loop(total_generations, games_per_generation):
    """
    Runs the main generational training loop.
    """
    for gen in range(total_generations):
        print(f"=============== STARTING GENERATION {gen} ===============")

        # 1. Select and prepare the agent for simulation
        if gen == 0:
            print("Using RandomAgent for initial data generation.")
            agent = RandomAgent()
        else:
            model_path_prefix = f"models/generation_{gen}/pandemic_model"
            print(f"Loading NNAgent from {model_path_prefix}...")
            agent = NNAgent(model_path_prefix)
            if not agent.model:
                print("Failed to load model. Halting evolution.")
                break
        
        # 2. Run the simulation to generate data for this generation
        sim_output_path = f"data/generation_{gen}/simulation_data.json"
        run_simulation(agent, games_per_generation, sim_output_path)

        # 3. Analyze the results of the simulation
        analyze_generation_data(sim_output_path)

        # 4. Train the next generation's model based on this data
        print(f"Training model for Generation {gen + 1}...")
        train_next_generation(gen)
        
        print(f"=============== FINISHED GENERATION {gen} ===============\n")


if __name__ == "__main__":
    # --- Configuration ---
    TOTAL_GENERATIONS = 5  # The number of generations to evolve
    GAMES_PER_GENERATION = 1000 # The number of games to play in each generation

    main_loop(TOTAL_GENERATIONS, GAMES_PER_GENERATION)
