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
    input_dim = env.get_node_feature_count()
    agent = GNNAgent(input_dim=input_dim, config=config)
    
    try:
        agent.load_model(playback_cfg['model_path'])
        agent.policy_network.eval()
    except FileNotFoundError:
        log(f"ERROR: Model not found at {playback_cfg['model_path']}. Cannot run playback.")
        return

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
        cures_found = [color for color, found in env.cures.items() if found]
        cubes_data = {city: data['cubes'] for city, data in env.board_state.items()}
        non_zero_cubes = {city: {c:v for c,v in cubes.items() if v > 0} for city, cubes in cubes_data.items()}
        non_zero_cubes = {city: cubes for city, cubes in non_zero_cubes.items() if cubes}
        
        log(f"Player is at: {loc}")
        log(f"Cures Found: {cures_found if cures_found else 'None'}")
        log(f"Investigation Centers: {env.investigation_centers if env.investigation_centers else 'None'}")
        log(f"Cubes on board: {non_zero_cubes if non_zero_cubes else 'None'}")
        log(f"Player Hand: {env.player_hand}")

        with torch.no_grad():
            state.batch = torch.zeros(state.num_nodes, dtype=torch.long)
            log("\nModel Evaluation:")
            node_embeddings, graph_embedding = agent.policy_network(state)
            
            header = (f"{'City':<15} | {'Cubes(B,Y,K,R)':<16} | {'Cures(B,Y,K,R)':<16} | "
                      f"{'Hand(B,Y,K,R)':<16} | {'Player':>6} | {'HasCard':>7} | {'Center':>6}")
            log(header)
            log('-' * len(header))

            for i, city in enumerate(env.all_cities):
                features = state.x[i].numpy()
                cubes_str = f"[{features[0]:.1f} {features[1]:.1f} {features[2]:.1f} {features[3]:.1f}]"
                cures_str = f"[{int(features[4])} {int(features[5])} {int(features[6])} {int(features[7])}]"
                hand_str = f"[{features[8]:.1f} {features[9]:.1f} {features[10]:.1f} {features[11]:.1f}]"
                player_str = f"{int(features[12])}"
                has_card_str = f"{int(features[13])}"
                has_center_str = f"{int(features[14])}"

                row = (f"{city:<15} | {cubes_str:<16} | {cures_str:<16} | "
                       f"{hand_str:<16} | {player_str:>6} | {has_card_str:>7} | {has_center_str:>6}")
                log(row)

            log("\nAction Scores:")
            possible_actions_mask = env.get_possible_action_mask()
            logits = torch.full_like(possible_actions_mask, -1e8, dtype=torch.float)
            
            for action_idx, is_possible in enumerate(possible_actions_mask):
                if is_possible:
                    action_desc = env.idx_to_action[action_idx]
                    score = 0
                    if action_desc.get('type') == 'move':
                        score = agent.policy_network.move_head(node_embeddings[action_desc['target_idx']])
                    elif action_desc.get('type') == 'treat':
                        score = agent.policy_network.treat_head(node_embeddings[action_desc['target_idx']])
                    elif action_desc.get('type') == 'discover_cure':
                        color_scores = agent.policy_network.cure_head(graph_embedding)
                        color_idx = agent.colors.index(action_desc['color'])
                        score = color_scores[0, color_idx]
                    elif action_desc.get('type') == 'build_investigation_center':
                        score = agent.policy_network.build_head(node_embeddings[action_desc['target_idx']])
                    elif action_desc.get('type') == 'pass':
                        score = agent.policy_network.pass_head(graph_embedding)
                    
                    logits[action_idx] = score
                    log(f"  - Action: {action_desc}, Raw Score: {score.item():.4f}")
            
            prob_dist = Categorical(logits=logits)
            chosen_action_idx = prob_dist.sample()
            chosen_action_desc = env.idx_to_action[chosen_action_idx.item()]

        log(f"\n==> Agent chose: {chosen_action_desc}")
        state, _, done = env.step(chosen_action_idx.item())

    log("\n=============== PLAYBACK FINISHED ===============")
    log(f"Game Over. Final Result: {env.is_game_over()[1]}")
    log(f"Total Actions: {env.actions_taken}")