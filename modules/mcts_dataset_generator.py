# modules/mcts_dataset_generator.py

import random
import json
import torch
import os
from collections import Counter
from game.pandemic_game import PandemicGame
from agents.agents import GNN_MCTS_Agent, RandomAgent

# --- MODIFIED: The function now takes an output_path and saves data instead of returning it ---
def generate_training_data(agent_class, model_path, difficulties, num_games, config, output_path):
    """
    Plays a set of games and saves the generated training data to a file.
    """
    training_data = []
    win_count = 0
    
    print(f"--- Starting data generation: {num_games} games on difficulties {difficulties} ---")

    for i in range(num_games):
        difficulty = random.choice(difficulties)
        env = PandemicGame(difficulty=difficulty, config=config)

        agent = None
        if agent_class == GNN_MCTS_Agent:
            if model_path:
                try:
                    agent = GNN_MCTS_Agent(model_path, difficulty, config)
                except FileNotFoundError:
                    agent = RandomAgent()
            else:
                agent = RandomAgent()
        else:
            agent = RandomAgent()

        game_state_history = []
        done = False
        game_state_history.append(env.get_state_as_graph())

        while not done:
            possible_actions = env.get_possible_actions()
            if not possible_actions: break
            chosen_action = agent.choose_action(env, possible_actions)
            chosen_action_json = json.dumps(chosen_action, sort_keys=True)
            chosen_action_idx = -1
            for idx, action_dict in env.idx_to_action.items():
                if json.dumps(action_dict, sort_keys=True) == chosen_action_json:
                    chosen_action_idx = idx
                    break
            if chosen_action_idx == -1: break
            new_state, _, done = env.step(chosen_action_idx)
            game_state_history.append(new_state)

        _, result = env.is_game_over()
        target_value = 0.0
        if result == "win":
            win_count += 1
            base_win_reward = 10000.0
            final_win_reward = base_win_reward - (1.0 * env.actions_taken)
            target_value = max(0, final_win_reward)
        elif result == "loss":
            target_value = -9800.0

        for state_graph in game_state_history:
            training_data.append((state_graph, torch.tensor([target_value], dtype=torch.float32)))

        if (i + 1) % 50 == 0:
            current_win_rate = (win_count / (i + 1)) * 100
            # print(f"  ...generated data for {i+1}/{num_games} games. Current Win Rate: {current_win_rate:.1f}%")

    final_win_rate = (win_count / num_games) * 100 if num_games > 0 else 0
    # Save the generated data to the specified file path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(training_data, output_path)
    
    print(f"--- Data generation finished. Saved {len(training_data)} examples to {output_path}. Final Win Rate: {final_win_rate:.1f}% ---")