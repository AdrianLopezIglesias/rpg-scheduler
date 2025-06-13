import os
import shutil
import json
from ..simulation_runner import run_simulation
from ..trainer import train_model_on_data
from ..analysis import analyze_generation_data
from ..utils import log
from agents.agents import RandomAgent, NNAgent

def run_training_loop(config):
    """
    Manages a gated training curriculum.
    For each difficulty, it runs generations until a target win rate is met
    or max generations are reached.
    """
    train_cfg = config['train_config']
    curriculum_targets = train_cfg['curriculum']
    max_gens_per_difficulty = train_cfg['max_generations_per_difficulty']

    log("--- Cleaning up data and models for a fresh run ---")
    if os.path.exists("data"):
        shutil.rmtree("data")
    if os.path.exists("models"):
        shutil.rmtree("models")
    os.makedirs("models", exist_ok=True)
    
    best_model_from_previous_difficulty = None
    
    for difficulty, target_win_rate in curriculum_targets.items():
        log(f"=============== STARTING DIFFICULTY: {difficulty.upper()} ===============")
        log(f"  -> Target Win Rate: {target_win_rate}%")
        
        last_model_for_this_difficulty = best_model_from_previous_difficulty

        for gen in range(max_gens_per_difficulty):
            log(f"--- Generation {gen + 1}/{max_gens_per_difficulty} for '{difficulty}' ---")
            
            agent_to_use = RandomAgent()
            if last_model_for_this_difficulty:
                log(f"  -> Loading model: {last_model_for_this_difficulty}")
                agent_to_use = NNAgent(last_model_for_this_difficulty, epsilon=0.2)
            else:
                log("  -> No previous model found. Using RandomAgent.")
            
            sim_output_path = f"data/{difficulty}/generation_{gen}/simulation_data.json"
            log(f"  -> Running simulation with {type(agent_to_use).__name__}...")
            run_simulation(agent_to_use, train_cfg['games_per_generation'], sim_output_path, difficulty, config)

            with open(sim_output_path, 'r') as f:
                analysis_results = analyze_generation_data(json.load(f))
            current_win_rate = analysis_results.get("win_rate_percent", 0)

            if current_win_rate >= target_win_rate:
                log(f"  SUCCESS: Met target win rate for '{difficulty}'. Moving to next difficulty.")
                best_model_from_previous_difficulty = last_model_for_this_difficulty
                break

            log(f"  -> Win rate of {current_win_rate:.2f}% is below target of {target_win_rate}%. Training new model...")
            candidate_model_path = f"models/{difficulty}/candidate_gen_{gen}"
            train_model_on_data(gen, difficulty, config, candidate_model_path)

            if os.path.exists(f"{candidate_model_path}.joblib"):
                last_model_for_this_difficulty = candidate_model_path
            else:
                log("  -> Training did not produce a new model. Reusing previous model for next generation.")
        
        else:
            log(f"FAILURE: Could not meet target for '{difficulty}' within {max_gens_per_difficulty} generations. Stopping training.")
            return

    log("=============== TRAINING CURRICULUM FINISHED ===============")