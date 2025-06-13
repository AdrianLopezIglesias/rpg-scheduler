import torch
from game.pandemic_game import PandemicGame
from agents.agents import GNNAgent
from .utils import log

def run_gnn_playback(config):
    playback_cfg = config['playback_config']
    log(f"=============== GNN AGENT PLAYBACK ===============")
    log(f"Loading model from: {playback_cfg['model_path']}")
    log(f"Playing on '{playback_cfg['difficulty']}' map.")
    env = PandemicGame(difficulty=playback_cfg['difficulty'], config=config)
    input_dim = 2
    output_dim = env.get_action_space_size()
    agent = GNNAgent(input_dim=input_dim, output_dim=output_dim, config=config)
    try:
        agent.load_model(playback_cfg['model_path'])
        agent.policy_network.eval()
    except FileNotFoundError:
        log(f"ERROR: Model not found at {playback_cfg['model_path']}. Cannot run playback.")
        return
    state = env.reset()
    done = False
    turn = 0
    while not done:
        turn += 1
        log(f"\n--- Turn {turn} | Actions Taken: {env.actions_taken} ---")
        loc = env.player_location
        cubes = {city: data['cubes'] for city, data in env.board_state.items() if data['cubes'] > 0}
        log(f"Player is at: {loc}")
        log(f"Cubes on board: {cubes if cubes else 'None'}")
        with torch.no_grad():
            state.batch = torch.zeros(state.num_nodes, dtype=torch.long)
            logits = agent.policy_network(state).squeeze(0)
            action_mask = env.get_possible_action_mask()
            logits[~action_mask] = -float('inf')
            probs = torch.softmax(logits, dim=-1)
            chosen_action_idx = agent.choose_action(state, action_mask)
            chosen_action_desc = env.idx_to_action[chosen_action_idx]
        log("Model's Top 5 Action Probabilities:")
        top_probs, top_indices = torch.topk(probs, 5)
        for i in range(len(top_probs)):
            idx = top_indices[i].item()
            prob = top_probs[i].item()
            if prob > 0:
                action_desc = env.idx_to_action[idx]
                log(f"  - Action: {action_desc}, Probability: {prob:.4f}")
        log(f"==> Agent chose: {chosen_action_desc}")
        state, _, done = env.step(chosen_action_idx)
    log("\n=============== PLAYBACK FINISHED ===============")
    log(f"Game Over. Final Result: {env.is_game_over()[1]}")
    log(f"Total Actions: {env.actions_taken}")