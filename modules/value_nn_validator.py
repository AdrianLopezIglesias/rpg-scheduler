import torch
import numpy as np
import json
from game.pandemic_game_extended import PandemicGameExtended
from agents.value_nn_agent import ValueNNAgent
from .utils import log

def run_value_nn_validation(config, model_path, difficulty):
    """
    Runs a validation test for the ValueNNAgent.
    """
    val_cfg = config.get('value_nn_curriculum_config', {})
    num_games = val_cfg.get('validation_games', 20)

    log("=============== RUNNING VALUE NN VALIDATION ===============")
    log(f"Model: {model_path}")
    log(f"Map: '{difficulty}'")
    log(f"Games: {num_games}")

    env = PandemicGameExtended(difficulty=difficulty, config=config)
    input_dim = env.get_feature_vector_size()

    try:
        agent = ValueNNAgent(
            model_path=model_path,
            input_dim=input_dim,
            config=config
        )
    except FileNotFoundError:
        log(f"ERROR: Model not found at {model_path}. Cannot run validation.")
        return {"win_rate_percent": 0, "fastest_win_actions": 'N/A', "avg_win_speed": 'N/A'}

    win_count = 0
    win_actions = []

    for i in range(num_games):
        env.reset()
        done = False
        
        with torch.no_grad():
            while not done:
                possible_actions_mask = env.get_possible_action_mask()
                if not any(possible_actions_mask):
                    break
                
                chosen_action_idx = agent.choose_action(env, possible_actions_mask)
                
                if chosen_action_idx == -1:
                    log("Error: Agent chose an invalid action.")
                    break
                
                _, _, done = env.step(chosen_action_idx)
                
                is_game_over, _ = env.is_game_over()
                if is_game_over:
                    done = True
        
        result = env.is_game_over()[1]
        if result == "win":
            win_count += 1
            win_actions.append(env.actions_taken)
        
        if (i + 1) % 10 == 0:
            log(f"  ...completed {i+1}/{num_games} validation games.")

    win_rate = (win_count / num_games) * 100 if num_games > 0 else 0
    fastest_win = min(win_actions) if win_actions else "N/A"
    avg_win_speed = np.mean(win_actions) if win_actions else "N/A"

    log("\n=============== VALUE NN VALIDATION RESULTS ===============")
    log(f"Win Rate: {win_rate:.2f}% ({win_count}/{num_games})")
    log(f"Fastest Win: {fastest_win} actions")
    avg_win_speed_str = f"{avg_win_speed:.2f}" if isinstance(avg_win_speed, float) else "N/A"
    log(f"Average Win Speed: {avg_win_speed_str} actions")
    log("=========================================================")
    
    return {"win_rate_percent": win_rate, "fastest_win_actions": fastest_win, "avg_win_speed": avg_win_speed}
