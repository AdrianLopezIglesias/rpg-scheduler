import torch
from torch.distributions import Categorical
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
    agent = GNNAgent(input_dim=input_dim, config=config)
    
    try:
        agent.load_model(playback_cfg['model_path'])
        agent.policy_network.eval()
    except FileNotFoundError:
        log(f"ERROR: Model not found at {playback_cfg['model_path']}. Cannot run playback.")
        return

    # Log map information
    log("\n--- Map Layout ---")
    for city, data in env.map.items():
        log(f"  - {city}: connected to {data['neighbors']}")
        
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
            
            # --- Detailed evaluation logging ---
            log("Model Evaluation:")
            node_embeddings, graph_embedding = agent.policy_network(state)
            
            # Log node features that the GNN sees
            log("  Node Features (Cubes, IsPlayer):")
            for i, city in enumerate(env.all_cities):
                log(f"    {city:<15}: {state.x[i].numpy()}")

            # Calculate and log scores for all legal actions
            log("  Action Scores:")
            possible_actions_mask = env.get_possible_action_mask()
            logits = torch.full_like(possible_actions_mask, -1e8, dtype=torch.float)
            
            for action_idx, is_possible in enumerate(possible_actions_mask):
                if is_possible:
                    action_desc = env.idx_to_action[action_idx]
                    score = 0
                    if action_desc['type'] == 'move':
                        target_node_idx = action_desc['target_idx']
                        score = agent.policy_network.move_head(node_embeddings[target_node_idx])
                    elif action_desc['type'] == 'treat':
                        target_node_idx = action_desc['target_idx']
                        score = agent.policy_network.treat_head(node_embeddings[target_node_idx])
                    elif action_desc['type'] == 'pass':
                        score = agent.policy_network.pass_head(graph_embedding)
                    
                    logits[action_idx] = score
                    log(f"    - Action: {action_desc}, Raw Score: {score.item():.4f}")
            
            # Get final choice
            prob_dist = Categorical(logits=logits)
            chosen_action_idx = prob_dist.sample()
            chosen_action_desc = env.idx_to_action[chosen_action_idx.item()]
            # -----------------------------------

        log(f"==> Agent chose: {chosen_action_desc}")
        state, _, done = env.step(chosen_action_idx.item())

    log("\n=============== PLAYBACK FINISHED ===============")
    log(f"Game Over. Final Result: {env.is_game_over()[1]}")
    log(f"Total Actions: {env.actions_taken}")