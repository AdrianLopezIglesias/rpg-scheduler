import os
import json
import numpy as np
import time
import argparse
from datetime import datetime

from modules.simulation_runner import run_simulation
from modules.trainer import train_next_generation
from agents.agents import RandomAgent, NNAgent, create_feature_vector
from game.pandemic_game import PandemicGame

def log(message):
    """Prints a message with a timestamp."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def analyze_generation_data(list_of_games):
    """Analyzes game data and returns a dictionary of performance metrics."""
    num_games = len(list_of_games)
    winning_games = [g for g in list_of_games if g.get("final_result") == "win"]
    win_rate = len(winning_games) / num_games * 100 if num_games > 0 else 0
    
    if winning_games:
        win_lengths = [g["total_actions"] for g in winning_games]
        avg_win_speed = np.mean(win_lengths)
        fastest_win = min(win_lengths)
    else:
        avg_win_speed = "N/A"
        fastest_win = "N/A"

    analysis = {
        "win_rate_percent": win_rate,
        "avg_actions_to_win": avg_win_speed,
        "fastest_win_actions": fastest_win
    }

    log("--- Generation Analysis ---")
    log(f"Win Rate: {analysis['win_rate_percent']:.2f}% | Avg. Win Speed: {analysis['avg_actions_to_win']} | Fastest Win: {analysis['fastest_win_actions']} actions")
    
    return analysis

def run_training_loop(config):
    """Runs the main generational training loop."""
    cfg = config['train_config']
    agent_cfg = config['agent_config']
    curriculum = cfg['curriculum']
    total_generations = len(curriculum)
    
    epsilon_decay = (agent_cfg['epsilon_start'] - agent_cfg['epsilon_end']) / total_generations if total_generations > 0 else 0

    for gen, difficulty in enumerate(curriculum):
        log(f"=============== STARTING GENERATION {gen} on '{difficulty}' map ===============")
        
        sim_output_path = f"data/{difficulty}/generation_{gen}/simulation_data.json"
        start_time = time.time()

        if gen == 0:
            agent = RandomAgent()
        else:
            prev_difficulty = curriculum[gen - 1]
            model_path_prefix = f"models/{prev_difficulty}/generation_{gen}/pandemic_model"
            current_epsilon = agent_cfg['epsilon_start'] - (gen * epsilon_decay)
            agent = NNAgent(model_path_prefix, epsilon=current_epsilon)
            if not agent.model:
                log(f"Failed to load model for Gen {gen}. Halting.")
                break
        
        run_simulation(agent, cfg['games_per_generation'], sim_output_path, difficulty=difficulty, config=config)
        log(f"Simulation finished in {time.time() - start_time:.2f}s.")
        
        analysis_results = {}
        try:
            with open(sim_output_path, 'r') as f:
                analysis_results = analyze_generation_data(json.load(f))
        except FileNotFoundError:
            log(f"Could not find simulation data at {sim_output_path} to analyze.")
        
        start_time = time.time()
        train_next_generation(gen, difficulty, config, analysis_results)
        log(f"Training finished in {time.time() - start_time:.2f}s.")

def run_test(config):
    """Runs a batch test of a specified model."""
    cfg = config['test_config']
    log(f"Testing Gen {cfg['model_generation_to_test']} ('{cfg['model_training_difficulty']}') on '{cfg['map_difficulty_to_test_on']}' map...")

    model_path = f"models/{cfg['model_training_difficulty']}/generation_{cfg['model_generation_to_test']}/pandemic_model"
    agent = NNAgent(model_path, epsilon=0) # Epsilon=0 for pure exploitation
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
    player_loc = state['player_location']
    board = state['board']
    
    log(f"--- Turn {state['actions_taken']} | Player at: {player_loc} ---")
    
    # --- Detailed Input Vector Breakdown ---
    print("  [Input Vector Breakdown]")
    print(f"    - Cubes at Current Location ({player_loc}): {board[player_loc]['cubes']}")
    neighbors1 = sorted(game.map[player_loc]['neighbors'])
    l1_strings = [f"{n}({board[n]['cubes']})" for n in neighbors1]
    print(f"    - L1 Neighbors (sorted): {', '.join(l1_strings)}")
    neighbors2_set = set(n2 for n1 in neighbors1 for n2 in game.map[n1]['neighbors'])
    neighbors2 = sorted(list(neighbors2_set - set(neighbors1) - {player_loc}))
    l2_strings = [f"{n}({board[n]['cubes']})" for n in neighbors2]
    print(f"    - L2 Neighbors: {', '.join(l2_strings)}")
    neighbors3_set = set(n3 for n2 in neighbors2 for n3 in game.map[n2]['neighbors'])
    neighbors3 = sorted(list(neighbors3_set - set(neighbors2) - set(neighbors1) - {player_loc}))
    l3_strings = [f"{n}({board[n]['cubes']})" for n in neighbors3]
    print(f"    - L3 Neighbors: {', '.join(l3_strings)}")
    total_cubes = sum(c['cubes'] for c in board.values())
    print(f"    - Total Cubes on Board: {total_cubes}")
    danger_cities = [city for city, data in board.items() if data['cubes'] == 3]
    print(f"    - Danger City Count: {len(danger_cities)}")
    dist_to_danger = min([game.get_distance(player_loc, c) for c in danger_cities]) if danger_cities else -1
    print(f"    - Distance to Nearest Danger: {dist_to_danger}")
    feature_vector = create_feature_vector(state, game)
    print(f"    - Full Vector: {feature_vector}")
    # --- End Breakdown ---
    
    # Show the agent's "thought process" if it's an NN agent
    if isinstance(agent, NNAgent) and agent.model:
        log("Evaluating possible actions...")
        possible_actions = game.get_possible_actions()
        
        for action in possible_actions:
            future_state = agent._simulate_next_state(game, action)
            future_feature_vector = create_feature_vector(future_state, game)
            scaled_features = agent.scaler.transform(future_feature_vector)
            predicted_score = agent.model.predict(scaled_features)[0]
            print(f"    - If action is {action}, predicted future state score: {predicted_score:.2f}")

    chosen_action = agent.choose_action(game, game.get_possible_actions())
    log(f"==> Agent's Final Decision: {chosen_action}\n")
    return chosen_action


def run_debug(config):
    """Runs a single game with detailed turn-by-turn debug output."""
    cfg = config['debug_config']
    log(f"Debugging Gen {cfg['model_generation_to_debug']} ('{cfg['model_training_difficulty']}') on '{cfg['map_difficulty_to_debug_on']}' map...")
    
    game = PandemicGame(difficulty=cfg['map_difficulty_to_debug_on'])
    model_path = f"models/{cfg['model_training_difficulty']}/generation_{cfg['model_generation_to_debug']}/pandemic_model"
    agent = NNAgent(model_path, epsilon=0)

    if not agent.model:
        log("Model not found, cannot run debug session.")
        return

    game.reset()
    while not game.is_game_over():
        action = debug_decision_process(game, agent)
        game.step(action)
    
    log(f"Debug game finished. Result: {game.is_game_over()}, Total Actions: {game.actions_taken}")


def main():
    parser = argparse.ArgumentParser(description="Pandemic AI Training and Testing CLI")
    parser.add_argument("command", choices=["train", "test", "debug"], help="The action to perform.")
    args = parser.parse_args()

    try:
        with open("config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        log("Error: config.json not found. Please create it.")
        return

    if args.command == "train":
        run_training_loop(config)
    elif args.command == "test":
        run_test(config)
    elif args.command == "debug":
        run_debug(config)

if __name__ == "__main__":
    main()
