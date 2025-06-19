import torch
import numpy as np
import json
from game.pandemic_game import PandemicGame
# --- MODIFIED: Import the new GNN_MCTS_Agent ---
from agents.agents import GNN_MCTS_Agent
from .utils import log

def run_validation(config):
    val_cfg = config['validation_config']
    log("=============== RUNNING MODEL VALIDATION ===============")
    log(f"Model: {val_cfg['model_path']}")
    log(f"Map: '{val_cfg['difficulty']}'")
    log(f"Games: {val_cfg['num_games']}")

    env = PandemicGame(difficulty=val_cfg['difficulty'], config=config)

    # --- MODIFIED: Instantiate the GNN_MCTS_Agent ---
    try:
        agent = GNN_MCTS_Agent(
            model_path=val_cfg['model_path'], 
            difficulty=val_cfg['difficulty'], 
            config=config
        )
    except FileNotFoundError:
        log(f"ERROR: Model not found at {val_cfg['model_path']}. Cannot run validation.")
        return {"win_rate_percent": 0, "fastest_win_actions": 'N/A', "avg_win_speed": 'N/A'}

    win_count = 0
    win_actions = []

    for i in range(val_cfg['num_games']):
        env.reset()
        done = False
        
        # --- REWRITTEN: The validation loop now uses the new agent's interface ---
        with torch.no_grad():
            while not done:
                # 1. Get possible actions as a list
                possible_actions = env.get_possible_actions()
                
                # 2. Agent chooses the best action based on its internal lookahead
                chosen_action = agent.choose_action(env, possible_actions)
                
                # 3. Convert the chosen action dictionary back to an index for the step function
                chosen_action_json = json.dumps(chosen_action, sort_keys=True)
                chosen_action_idx = -1
                for idx, action_dict in env.idx_to_action.items():
                    if json.dumps(action_dict, sort_keys=True) == chosen_action_json:
                        chosen_action_idx = idx
                        break
                
                if chosen_action_idx == -1:
                    log("Error: Agent chose an action that could not be mapped to an index.")
                    break
                
                # 4. Step the environment
                _, _, done = env.step(chosen_action_idx)
                
                is_game_over, _ = env.is_game_over()
                if is_game_over:
                    done = True
        
        result = env.is_game_over()[1]
        if result == "win":
            win_count += 1
            win_actions.append(env.actions_taken)

    win_rate = (win_count / val_cfg['num_games']) * 100 if val_cfg['num_games'] > 0 else 0
    fastest_win = min(win_actions) if win_actions else "N/A"
    avg_win_speed = np.mean(win_actions) if win_actions else "N/A"

    log("\n=============== 🧱🧱🧱🧱 VALIDATION RESULTS 🧱🧱🧱🧱 ===============")
    log(f"Win Rate: {win_rate:.2f}% ({win_count}/{val_cfg['num_games']})")
    log(f"Fastest Win: {fastest_win} actions")
    log(f"Average Win Speed: {avg_win_speed:.2f} actions" if isinstance(avg_win_speed, float) else f"{avg_win_speed} actions")
    log("================================================")
    
    return {"win_rate_percent": win_rate, "fastest_win_actions": fastest_win, "avg_win_speed": avg_win_speed}