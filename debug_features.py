import os
import json
import numpy as np
import time
from datetime import datetime
from game.pandemic_game import PandemicGame
from agents.agents import RandomAgent, NNAgent, create_feature_vector

def log(message):
    """Prints a message with a timestamp."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def debug_decision_process(game, agent):
    """
    Analyzes and prints the step-by-step decision-making process of an NNAgent for a single turn.
    """
    state = game.get_state_snapshot()
    player_loc = state['player_location']
    possible_actions = game.get_possible_actions()
    board = state['board']

    log(f"--- Turn {state['actions_taken']} | Player at: {player_loc} ---")
    
    # --- Feature Vector Breakdown ---
    print("  [Input Vector Breakdown]")
    
    # 1. Local Features
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

    # 2. Global Features
    total_cubes = sum(c['cubes'] for c in board.values())
    print(f"    - Total Cubes on Board: {total_cubes}")
    
    danger_cities = [city for city, data in board.items() if data['cubes'] == 3]
    print(f"    - Danger City Count: {len(danger_cities)}")
    
    dist_to_danger = min([game.get_distance(player_loc, c) for c in danger_cities]) if danger_cities else -1
    print(f"    - Distance to Nearest Danger: {dist_to_danger}")
    
    # 3. If it's an NNAgent, show its internal "thoughts".
    if isinstance(agent, NNAgent) and agent.model:
        feature_vector = create_feature_vector(state, game)
        scaled_features = agent.scaler.transform(feature_vector)
        action_scores = agent.model.predict(scaled_features)[0]
        
        print("\n  [Model Decision Analysis]")
        log(f"Model Output (Raw Action Scores): {np.round(action_scores, 2)}")

        # Map the generic scores to the currently legal actions.
        sorted_neighbors = sorted(game.map[player_loc]['neighbors'])
        
        is_treat_legal = {"type": "treat", "target": player_loc} in possible_actions
        print(f"    - Treat at {player_loc}: Score={action_scores[0]:.2f} (Legal: {is_treat_legal})")

        for i in range(6): 
            action_score_index = i + 1
            if action_score_index < len(action_scores):
                score = action_scores[action_score_index]
                if i < len(sorted_neighbors):
                    neighbor = sorted_neighbors[i]
                    is_move_legal = {"type": "move", "target": neighbor} in possible_actions
                    print(f"    - Move to {neighbor} (Neighbor #{i+1}): Score={score:.2f} (Legal: {is_move_legal})")
    
    # 4. Show the final decision made by the agent.
    chosen_action = agent.choose_action(game, possible_actions)
    log(f"==> Agent's Final Decision: {chosen_action}\n")
    return chosen_action


def get_latest_generation(models_dir):
    """Finds the highest generation number in a models directory."""
    if not os.path.isdir(models_dir):
        return None
    
    gen_dirs = [d for d in os.listdir(models_dir) if d.startswith('generation_')]
    if not gen_dirs:
        return None
        
    gen_numbers = [int(d.split('_')[1]) for d in gen_dirs]
    return max(gen_numbers)


def run_debug_game(model_gen, train_difficulty, test_difficulty):
    """
    Loads a specific agent and runs a single game, printing the debug
    information for each turn.
    """
    log(f"Setting up debug game for model Gen {model_gen} ('{train_difficulty}') on '{test_difficulty}' map.")
    
    game = PandemicGame(difficulty=test_difficulty)
    model_path_prefix = f"models/{train_difficulty}/generation_{model_gen}/pandemic_model"
    agent = NNAgent(model_path_prefix, epsilon=0) # Epsilon=0 for pure exploitation

    if not agent.model:
        log("Could not load NNAgent, using RandomAgent for debugging.")
        agent = RandomAgent()

    game.reset()
    
    while not game.is_game_over():
        # The debug function now returns the action, so we don't need to call agent.choose_action() twice.
        action = debug_decision_process(game, agent)
        game.step(action)
    
    log(f"Debug game finished. Final Result: {game.is_game_over()}, Total Actions: {game.actions_taken}")


if __name__ == "__main__":
    # --- Configuration ---
    MODEL_TRAINING_DIFFICULTY = "easy"
    DIFFICULTY_TO_TEST_ON = "hard"

    # --- Automatically find the latest model ---
    models_directory = f"models/{MODEL_TRAINING_DIFFICULTY}"
    latest_gen = get_latest_generation(models_directory)

    if latest_gen is not None:
        log(f"Found latest model: Generation {latest_gen} trained on '{MODEL_TRAINING_DIFFICULTY}'.")
        run_debug_game(
            latest_gen,
            MODEL_TRAINING_DIFFICULTY,
            DIFFICULTY_TO_TEST_ON
        )
    else:
        log(f"No trained models found in '{models_directory}'. Cannot run debug session.")
