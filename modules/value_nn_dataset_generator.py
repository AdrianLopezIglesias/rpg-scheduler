import torch
import os
import random
import json
from .utils import log
from agents.agents import RandomAgent
from agents.value_nn_agent import ValueNNAgent
from game.pandemic_game_extended import PandemicGameExtended

def generate_value_nn_data(agent_class, model_path, input_dim, difficulties, num_games, config, output_path):
    """
    Plays games and generates a balanced set of (state_vector, score) pairs
    for training the ValueNet.
    """
    
    all_played_games = []
    
    log(f"  -> Starting data generation: {num_games} games on difficulties {difficulties}...")

    for i in range(num_games):
        difficulty = random.choice(difficulties)
        env = PandemicGameExtended(difficulty=difficulty, config=config)

        agent = agent_class(model_path=model_path, input_dim=input_dim, config=config) if agent_class == ValueNNAgent else RandomAgent()

        game_state_history = [env.get_state_as_vector()]
        done = False

        while not done:
            possible_actions = env.get_possible_actions()
            if not possible_actions: 
                log("    ⚠️ No possible actions found. Breaking game loop.")
                break
            
            action_arg = env.get_possible_action_mask() if isinstance(agent, ValueNNAgent) else possible_actions
                
            chosen_action = agent.choose_action(env, action_arg)

            if chosen_action is None:
                log("    ⚠️ Agent returned a 'None' action. Breaking game loop.")
                break

            action_idx = -1
            if isinstance(chosen_action, dict):
                 action_json_str = json.dumps(chosen_action, sort_keys=True)
                 action_idx = env.action_to_idx.get(action_json_str, -1)
            else:
                 action_idx = chosen_action

            if action_idx == -1:
                log(f"    ⚠️ Could not map chosen action to a valid index. Breaking game loop. Action: {chosen_action}")
                break
            
            _, _, done = env.step(action_idx)
            game_state_history.append(env.get_state_as_vector())
        
        _, result = env.is_game_over()
        all_played_games.append({"history": game_state_history, "result": result})

        if (i + 1) % (num_games // 4 if num_games >= 4 else 1) == 0 and num_games > 10:
            log(f"    ...played {i+1}/{num_games} games.")
    
    log(f"  -> Game simulation finished. Total games played: {len(all_played_games)}.")
    
    log("  -> Balancing dataset...")
    winning_games = [g for g in all_played_games if str(g.get("result", "")).strip() == "win"]
    losing_games = [g for g in all_played_games if str(g.get("result", "")).strip() == "loss"]
    log(f"    Found {len(winning_games)} wins and {len(losing_games)} losses.")

    # Shuffle both lists to use a random sample of games, not just the most recent ones.
    random.shuffle(winning_games)
    random.shuffle(losing_games)
    
    num_to_keep = len(winning_games)
    selected_losing_games = losing_games[:num_to_keep]
    log(f"    Using all {len(winning_games)} wins and a random sample of {len(selected_losing_games)} losses for training.")

    balanced_games = winning_games + selected_losing_games
    random.shuffle(balanced_games)

    training_data = []
    total_game_rewards = []
    win_count = 0

    for game in balanced_games:
        num_steps = len(game["history"])
        max_reward = 1000.0
        
        is_win = str(game.get("result", "")).strip() == "win"
        R = max_reward if is_win else -max_reward
        total_game_rewards.append(R)
        
        if is_win:
            win_count += 1
            
        for turn, state_vector in enumerate(game["history"]):
            score = R - (R / num_steps) * turn if num_steps > 0 else R
            training_data.append((state_vector, torch.tensor([score], dtype=torch.float32)))

    log("  -> Saving data...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(training_data, output_path)

    avg_reward = sum(total_game_rewards) / len(total_game_rewards) if total_game_rewards else 0
    final_win_rate = (win_count / len(balanced_games)) * 100 if balanced_games else 0
    
    log(f"  -> Data generation finished. Using {len(balanced_games)} balanced games.")
    log(f"     Total steps saved to '{os.path.basename(output_path)}': {len(training_data)}")
    
    return {
        "win_rate": final_win_rate,
        "avg_reward": avg_reward
    }
