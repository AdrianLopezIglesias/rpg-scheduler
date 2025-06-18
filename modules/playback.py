import torch
import json
from torch.distributions import Categorical
from game.pandemic_game import PandemicGame
# --- MODIFIED: Import the new agent ---
from agents.agents import GNN_MCTS_Agent
from .utils import log
from collections import Counter

def run_gnn_playback(config):
    playback_cfg = config['playback_config']
    log(f"=============== GNN AGENT PLAYBACK ===============")
    log(f"Loading model from: {playback_cfg['model_path']}")
    log(f"Playing on '{playback_cfg['difficulty']}' map.")

    env = PandemicGame(difficulty=playback_cfg['difficulty'], config=config)
    
    # --- MODIFIED: Instantiate the new GNN_MCTS_Agent ---
    try:
        agent = GNN_MCTS_Agent(
            model_path=playback_cfg['model_path'], 
            difficulty=playback_cfg['difficulty'], 
            config=config
        )
    except FileNotFoundError:
        log(f"ERROR: Model not found at {playback_cfg['model_path']}. Cannot run playback.")
        return

    log("\n--- Map Layout ---")
    for city, data in env.map.items():
        log(f"  - {city}: connected to {data['neighbors']}")

    # The main loop is now much simpler
    done = False
    turn = 0
    while not done:
        turn += 1
        log(f"\n--- Turn {turn} | Actions Taken: {env.actions_taken} ---")

        # --- Logging logic remains the same ---
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

        # --- REWRITTEN: The new agent handles all evaluation internally ---
        log("\n--- Agent is thinking (simulating outcomes)... ---")
        possible_actions = env.get_possible_actions()
        chosen_action = agent.choose_action(env, possible_actions)
        
        # The agent returns the action dictionary. We need to find its index to step the environment.
        chosen_action_json = json.dumps(chosen_action, sort_keys=True)
        chosen_action_idx = -1
        for idx, action_dict in env.idx_to_action.items():
            if json.dumps(action_dict, sort_keys=True) == chosen_action_json:
                chosen_action_idx = idx
                break

        log(f"\n==> Agent chose: {chosen_action}")
        
        if chosen_action_idx != -1:
            _, _, done = env.step(chosen_action_idx)
        else:
            log("Error: Chosen action not found in action map.")
            break
            
        is_game_over, _ = env.is_game_over()
        if is_game_over:
            done = True

    log("\n=============== PLAYBACK FINISHED ===============")
    log(f"Game Over. Final Result: {env.is_game_over()[1]}")
    log(f"Total Actions: {env.actions_taken}")