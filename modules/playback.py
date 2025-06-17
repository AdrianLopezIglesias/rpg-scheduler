import torch
from torch.distributions import Categorical
from game.pandemic_game import PandemicGame
from agents.agents import GNNAgent
from .utils import log
from collections import Counter

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
        log(f"Player is at: {loc}")
        log(f"Investigation Centers: {env.investigation_centers if env.investigation_centers else 'None'}")
        cubes_data = {city: data['cubes'] for city, data in env.board_state.items()}
        non_zero_cubes = {city: {c:v for c,v in cubes.items() if v > 0} for city, cubes in cubes_data.items()}
        non_zero_cubes = {city: cubes for city, cubes in non_zero_cubes.items() if cubes}
        log(f"Cubes on board: {non_zero_cubes if non_zero_cubes else 'None'}")

        log("\n--- Global State ---")
        log(f"Player Hand: {env.player_hand}")
        log(f"Cards needed for cure: {env.cards_for_cure}")

        disease_statuses = {d['color']: d['status'] for d in env.diseases}
        cured_diseases = [color for color, status in disease_statuses.items() if status in ['cured', 'eradicated']]
        eradicated_diseases = [color for color, status in disease_statuses.items() if status == 'eradicated']
        log(f"Cured Diseases: {cured_diseases if cured_diseases else 'None'}")
        log(f"Eradicated Diseases: {eradicated_diseases if eradicated_diseases else 'None'}")
        
        actions_until_infection = 4 - (env.actions_taken % 4)
        log(f"Actions until next infection: {actions_until_infection}")
        deck_status = f"Player deck: {len(env.deck)} / {env.initial_deck_size} cards remaining"
        log(deck_status)
        log("Card Distribution:")
        cards_in_deck_by_color = Counter(env.map[card]['color'] for card in env.deck)
        cards_in_hand_by_color = Counter(env.map[card]['color'] for card in env.player_hand)
        for color in env.colors_in_play:
            total = env.total_cards_by_color.get(color, 0)
            in_deck = cards_in_deck_by_color.get(color, 0)
            in_hand = cards_in_hand_by_color.get(color, 0)
            discarded = total - in_deck - in_hand
            log(f"  - {color.capitalize()}: {in_deck} in deck, {in_hand} in hand, {discarded} discarded (Total: {total})")

        with torch.no_grad():
            state.batch = torch.zeros(state.num_nodes, dtype=torch.long)
            
            # --- MODIFIED: Unpack the state_value returned by the Actor-Critic network ---
            (node_embeddings, graph_embedding, _) = agent.policy_network(state)

            log("\n--- City-Specific State ---")
            header = (f"{'City':<15} | {'Cubes(B,Y,K,R)':<16} | {'Player':>6} | {'HasCard':>7} | {'Center':>6} | {'ShouldCenter':>12}")
            log(header)
            log('-' * len(header))
            
            for i, city in enumerate(env.all_cities):
                city_cubes = env.board_state[city]["cubes"]
                cubes_str = f"[{city_cubes['blue']} {city_cubes['yellow']} {city_cubes['black']} {city_cubes['red']}]"
                player_str = f"{1 if env.player_location == city else 0}"
                has_card_str = f"{1 if city in env.player_hand else 0}"
                has_center_str = f"{1 if city in env.investigation_centers else 0}"
                
                should_have_center_val = 0.0
                if city not in env.investigation_centers:
                    neighbor_has_center = any(neighbor in env.investigation_centers for neighbor in env.map[city]["neighbors"])
                    if not neighbor_has_center:
                        should_have_center_val = 1.0
                should_center_str = f"{int(should_have_center_val)}"

                row = (f"{city:<15} | {cubes_str:<16} | {player_str:>6} | {has_card_str:>7} | {has_center_str:>6} | {should_center_str:>12}")
                log(row)

            log("\nAction Scores:")
            possible_actions_mask = env.get_possible_action_mask()
            logits = torch.full_like(possible_actions_mask, -1e8, dtype=torch.float)

            for action_idx, is_possible in enumerate(possible_actions_mask):
                if is_possible:
                    action_desc = env.idx_to_action[action_idx]
                    score = 0
                    action_type = action_desc.get("type")

                    log(f"  - Action: {action_desc}")

                    if action_type == 'move':
                        target_node_embedding = node_embeddings[action_desc['target_idx']]
                        combined_embedding = torch.cat([target_node_embedding, graph_embedding.squeeze(0)])
                        # log(f"    Input to move_head: {combined_embedding.detach().numpy()}")
                        score = agent.policy_network.move_head(combined_embedding)
                    elif action_type == 'treat':
                        target_node_embedding = node_embeddings[action_desc['target_idx']]
                        combined_embedding = torch.cat([target_node_embedding, graph_embedding.squeeze(0)])
                        # log(f"    Input to treat_head: {combined_embedding.detach().numpy()}")
                        score = agent.policy_network.treat_head(combined_embedding)
                    elif action_type == 'discover_cure':
                        # log(f"    Input to cure_head: {graph_embedding.squeeze(0).detach().numpy()}")
                        color_scores = agent.policy_network.cure_head(graph_embedding)
                        color_idx = agent.colors.index(action_desc['color'])
                        score = color_scores[0, color_idx]
                    elif action_type == 'build_investigation_center':
                        target_node_embedding = node_embeddings[action_desc['target_idx']]
                        combined_embedding = torch.cat([target_node_embedding, graph_embedding.squeeze(0)])
                        # log(f"    Input to build_head: {combined_embedding.detach().numpy()}")
                        score = agent.policy_network.build_head(combined_embedding)
                    elif action_type == 'pass':
                        # log(f"    Input to pass_head: {graph_embedding.squeeze(0).detach().numpy()}")
                        score = agent.policy_network.pass_head(graph_embedding)

                    logits[action_idx] = score
                    log(f"    Raw Score: {score.item():.4f}")
            
            prob_dist = Categorical(logits=logits)
            chosen_action_idx = prob_dist.sample()
            chosen_action_desc = env.idx_to_action[chosen_action_idx.item()]

        log(f"\n==> Agent chose: {chosen_action_desc}")
        state, _, done = env.step(chosen_action_idx.item())

    log("\n=============== PLAYBACK FINISHED ===============")
    log(f"Game Over. Final Result: {env.is_game_over()[1]}")
    log(f"Total Actions: {env.actions_taken}")