import torch
import numpy as np
from game.pandemic_game import PandemicGame
from agents.agents import GNNAgent
from .utils import log

def run_validation(config):
    val_cfg = config['validation_config']
    log("=============== RUNNING MODEL VALIDATION ===============")
    log(f"Model: {val_cfg['model_path']}")
    log(f"Map: '{val_cfg['difficulty']}'")
    log(f"Games: {val_cfg['num_games']}")

    env = PandemicGame(difficulty=val_cfg['difficulty'], config=config)
    input_dim = env.get_node_feature_count()
    agent = GNNAgent(input_dim=input_dim, config=config)

    try:
        agent.load_model(val_cfg['model_path'])
        agent.policy_network.eval()
    except FileNotFoundError:
        log(f"ERROR: Model not found at {val_cfg['model_path']}. Cannot run validation.")
        return {"win_rate_percent": 0, "fastest_win_actions": 'N/A', "avg_win_speed": 'N/A'}

    win_count = 0
    win_actions = []

    for i in range(val_cfg['num_games']):
        state = env.reset()
        done = False
        with torch.no_grad():
            while not done:
                state.batch = torch.zeros(state.num_nodes, dtype=torch.long)
                action_idx = agent.choose_action(env, state)
                state, _, done = env.step(action_idx)
        
        result = env.is_game_over()[1]
        if result == "win":
            win_count += 1
            win_actions.append(env.actions_taken)

    win_rate = (win_count / val_cfg['num_games']) * 100
    fastest_win = min(win_actions) if win_actions else "N/A"
    avg_win_speed = np.mean(win_actions) if win_actions else "N/A"

    log("\n=============== VALIDATION RESULTS ===============")
    log(f"Win Rate: {win_rate:.2f}% ({win_count}/{val_cfg['num_games']})")
    log(f"Fastest Win: {fastest_win} actions")
    log(f"Average Win Speed: {avg_win_speed:.2f} actions" if isinstance(avg_win_speed, float) else f"{avg_win_speed} actions")
    log("================================================")
    
    return {"win_rate_percent": win_rate, "fastest_win_actions": fastest_win, "avg_win_speed": avg_win_speed}