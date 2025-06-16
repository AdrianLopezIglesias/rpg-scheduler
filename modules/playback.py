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
        cured_diseases = [d['color'] for d in env.diseases if d['status'] == 'cured']
        eradicated_diseases = [d['color'] for d in env.diseases if d['status'] == 'eradicated']
        cubes_data = {city: data['cubes'] for city, data in env.board_state.items()}
        non_zero_cubes = {city: {c:v for c,v in cubes.items() if v > 0} for city, cubes in cubes_data.items()}
        non_zero_cubes = {city: cubes for city, cubes in non_zero_cubes.items() if cubes}

        log(f"Player is at: {loc}")
        log(f"Cured Diseases: {cured_diseases if cured_diseases else 'None'}")
        log(f"Eradicated Diseases: {eradicated_diseases if eradicated_diseases else 'None'}")
        log(f"Investigation Centers: {env.investigation_centers if env.investigation_centers else 'None'}")
        log(f"Cubes on board: {non_zero_cubes if non_zero_cubes else 'None'}")
        log(f"Player Hand: {env.player_hand}")

        # --- NEW: Logging for global game state ---
        log("\n--- Global State ---")
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
        # --- End of new logging ---

        with torch.no_grad():
            state.batch = torch.zeros(state.num_nodes, dtype=torch.long)
            log("\nModel Evaluation:")
            node_embeddings, graph_embedding = agent.policy_network(state)

            header = (f"{'City':<15} | {'Cubes(B,Y,K,R)':<16} | {'Cures(B,Y,K,R)':<16} | {'Hand(B,Y,K,R)':<16} | {'Erad(B,Y,K,R)':<16} | {'Player':>6} | {'HasCard':>7} | {'Center':>6}")
            log(header)
            log('-' * len(header))

            hand_colors = Counter(env.map[card]['color'] for card in env.player_hand)
            disease_statuses = {d['color']: d['status'] for d in env.diseases}

            for i, city in enumerate(env.all_cities):
                city_cubes = env.board_state[city]["cubes"]

                cubes_str = f"[{city_cubes['blue']:.1f} {city_cubes['yellow']:.1f} {city_cubes['black']:.1f} {city_cubes['red']:.1f}]"
                cures_str = f"[{int(disease_statuses['blue'] in ['cured', 'eradicated'])} {int(disease_statuses['yellow'] in ['cured', 'eradicated'])} {int(disease_statuses['black'] in ['cured', 'eradicated'])} {int(disease_statuses['red'] in ['cured', 'eradicated'])}]"
                hand_str = f"[{hand_colors.get('blue', 0)/env.cards_for_cure:.1f} {hand_colors.get('yellow', 0)/env.cards_for_cure:.1f} {hand_colors.get('black', 0)/env.cards_for_cure:.1f} {hand_colors.get('red', 0)/env.cards_for_cure:.1f}]"
                erad_str = f"[{int(disease_statuses['blue'] == 'eradicated')} {int(disease_statuses['yellow'] == 'eradicated')} {int(disease_statuses['black'] == 'eradicated')} {int(disease_statuses['red'] == 'eradicated')}]"
                player_str = f"{1 if env.player_location == city else 0}"
                has_card_str = f"{1 if city in env.player_hand else 0}"
                has_center_str = f"{1 if city in env.investigation_centers else 0}"

                row = (f"{city:<15} | {cubes_str:<16} | {cures_str:<16} | "
                       f"{hand_str:<16} | {erad_str:<16} | {player_str:>6} | {has_card_str:>7} | {has_center_str:>6}")
                log(row)

            log("\nAction Scores:")
            possible_actions_mask = env.get_possible_action_mask()
            logits = torch.full_like(possible_actions_mask, -1e8, dtype=torch.float)

            for action_idx, is_possible in enumerate(possible_actions_mask):
                 if is_possible:
                    action_desc = env.idx_to_action[action_idx]
                    score = 0
                    action_type = action_desc.get("type")

                    if action_type == 'move':
                        target_node_embedding = node_embeddings[action_desc['target_idx']]
                        combined_embedding = torch.cat([target_node_embedding, graph_embedding.squeeze(0)])
                        score = agent.policy_network.move_head(combined_embedding)
                    elif action_type == 'treat':
                        target_node_embedding = node_embeddings[action_desc['target_idx']]
                        combined_embedding = torch.cat([target_node_embedding, graph_embedding.squeeze(0)])
                        score = agent.policy_network.treat_head(combined_embedding)
                    elif action_type == 'discover_cure':
                        color_scores = agent.policy_network.cure_head(graph_embedding)
                        color_idx = agent.colors.index(action_desc['color'])
                        score = color_scores[0, color_idx]
                    elif action_type == 'build_investigation_center':
                        score = agent.policy_network.build_head(node_embeddings[action_desc['target_idx']])
                    elif action_type == 'pass':
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