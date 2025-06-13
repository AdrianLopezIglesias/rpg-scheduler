import json
import numpy as np
import os
from .simulation_runner import run_simulation
from .analysis import analyze_generation_data
from .utils import log
from agents.agents import RandomAgent, NNAgent, create_feature_vector
from game.pandemic_game import PandemicGame

def validate_model(model_path_prefix, difficulty, config, target_win_rate):
    """
    Tests a model and returns a dictionary with analysis results.
    The dictionary includes a 'passed' key indicating if the win rate was met.
    """
    log(f"  Validating model from {model_path_prefix} on '{difficulty}' map...")
    cfg = config['champion_model_config']
    
    agent = NNAgent(model_path_prefix, epsilon=0)
    default_result = {"win_rate_percent": 0, "avg_actions_to_win": "N/A", "fastest_win_actions": "N/A", "passed": False}

    if not agent.model:
        return default_result
        
    validation_data_path = f"data/{difficulty}/validation_run.json"
    run_simulation(agent, cfg['games_to_validate'], validation_data_path, difficulty, config)
    
    try:
        with open(validation_data_path, 'r') as f:
            results = analyze_generation_data(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        return default_result
        
    results["passed"] = results.get("win_rate_percent", 0) >= target_win_rate
    return results

def run_test(config):
    """Runs a batch test of a specified model."""
    cfg = config['test_config']
    log(f"Testing Gen {cfg['model_generation_to_test']} ('{cfg['model_training_difficulty']}') on '{cfg['map_difficulty_to_test_on']}' map...")

    model_path = f"models/{cfg['model_training_difficulty']}/generation_{cfg['model_generation_to_test']}/pandemic_model"
    agent = NNAgent(model_path, epsilon=0)
    if not agent.model: 
        log("Model not found, cannot run test.")
        return

    output_path = f"data/test_runs/test_output.json"
    run_simulation(agent, cfg['num_games_to_test'], output_path, difficulty=cfg['map_difficulty_to_test_on'], config=config)
    
    with open(output_path, 'r') as f:
        analyze_generation_data(json.load(f))

def debug_decision_process(game, agent):
    """Analyzes and prints the agent's decision-making process for a single turn."""
    state = game.get_state_snapshot()
    log(f"--- Turn {state['actions_taken']} | Player at: {state['player_location']} ---")
    log(state)
    if isinstance(agent, NNAgent) and agent.model:
        log("Evaluating possible actions...")
        for action in game.get_possible_actions():
            future_state = agent._simulate_next_state(game, action)
            future_feature_vector = create_feature_vector(future_state, game)
            scaled_features = agent.scaler.transform(future_feature_vector)
            predicted_score = agent.model.predict(scaled_features)[0]
            print(f"  {action}, predicted future state score: {predicted_score:.2f}")

    chosen_action = agent.choose_action(game, game.get_possible_actions())
    log(f"==> Agent's Final Decision: {chosen_action}\n")
    return chosen_action

def run_debug(config):
    """Runs a single game with detailed turn-by-turn debug output."""
    cfg = config['debug_config']
    log(f"Debugging Champion ('{cfg['model_training_difficulty']}') on '{cfg['map_difficulty_to_debug_on']}' map...")
    
    game = PandemicGame(difficulty=cfg['map_difficulty_to_debug_on'], config=config)
    cfg = config['debug_config']
    candidate_model_path = f"models/{cfg['model_training_difficulty']}/candidate_gen_{cfg['model_generation_to_debug']}"
   
    agent = NNAgent(candidate_model_path, epsilon=0)

    if not agent.model:
        log("Champion model not found, cannot run debug session.")
        return

    game.reset()
    while not game.is_game_over():
        action = debug_decision_process(game, agent)
        game.step(action)
    
    log(f"Debug game finished. Result: {game.is_game_over()}, Total Actions: {game.actions_taken}")