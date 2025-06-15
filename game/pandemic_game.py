import random
import json
import os
from collections import deque, Counter
import torch
from torch_geometric.data import Data

class PandemicGame:
    def __init__(self, difficulty="0", config=None):
        if not config:
            raise ValueError("A config object must be provided to initialize the game.")

        self.difficulty = str(difficulty)
        self.all_possible_colors = ["blue", "yellow", "black", "red"]

        self.map_config = self._load_map_config(self.difficulty)
        self.colors_in_play = self.map_config["colors_in_play"]

        game_settings = config['game_settings']
        self.cards_for_cure = game_settings['cards_for_cure']
        self.map = self.map_config["cities"]
        self.max_actions_per_game = self.map_config.get("max_actions_per_game", 500)

        self.all_cities = list(self.map.keys())
        self.city_to_idx = {city: i for i, city in enumerate(self.all_cities)}
        self.idx_to_city = {i: city for city, i in self.city_to_idx.items()}

        self._build_action_maps()
        self._build_edge_index()
        self.reset()

    def _get_disease_status(self, color):
        for disease in self.diseases:
            if disease['color'] == color:
                return disease['status']
        return None

    def _is_disease_cured(self, color):
        status = self._get_disease_status(color)
        return status == 'cured' or status == 'eradicated'

    def _load_map_config(self, difficulty):
        config_path = os.path.join(os.path.dirname(__file__), 'maps.json')
        with open(config_path, 'r') as f:
            return json.load(f)[difficulty]

    def reset(self):
        self.player_location = random.choice(self.all_cities)
        self.actions_taken = 0
        self.outbreaks = 0

        self.deck = [city for city in self.all_cities]
        random.shuffle(self.deck)
        self.player_hand = []

        self.board_state = {city: {"cubes": {color: 0 for color in self.all_possible_colors}} for city in self.map}
        self.investigation_centers = set(self.map_config.get("investigation_centers", []))

        self.diseases = []
        cures_already_found = self.map_config.get("cures_found", [])
        for color in self.all_possible_colors:
            is_in_play = color in self.colors_in_play
            disease_status = 'not_in_play'
            if is_in_play:
                disease_status = 'cured' if color in cures_already_found else 'active'
            self.diseases.append({
                "color": color,
                "in_play": is_in_play,
                "status": disease_status
            })

        self.infection_deck = [city for city, data in self.map.items() if data['color'] in self.colors_in_play]
        random.shuffle(self.infection_deck)
        self.infection_discard = []

        self._setup_initial_board()
        for _ in range(3): self._draw_card()
        self._update_disease_statuses()
        return self.get_state_as_graph()

    def _setup_initial_board(self):
        infection_candidates = list(self.infection_deck)
        random.shuffle(infection_candidates)
        cards_to_infect = infection_candidates[:9]

        for city_name in cards_to_infect[0:3]:
            color = self.map[city_name]['color']
            self.board_state[city_name]['cubes'][color] = 3

        if len(cards_to_infect) > 3:
            for city_name in cards_to_infect[3:6]:
                color = self.map[city_name]['color']
                self.board_state[city_name]['cubes'][color] = 2

        if len(cards_to_infect) > 6:
            for city_name in cards_to_infect[6:9]:
                color = self.map[city_name]['color']
                self.board_state[city_name]['cubes'][color] = 1

        self.infection_discard.extend(cards_to_infect)
        for card in cards_to_infect:
             if card in self.infection_deck:
                 self.infection_deck.remove(card)

    def is_game_over(self):
        in_play_diseases = [d for d in self.diseases if d['in_play']]
        if all(d['status'] == 'eradicated' for d in in_play_diseases):
            return True, "win"

        if self.actions_taken >= self.max_actions_per_game or self.outbreaks >= 8:
            return True, "loss"

        return False, "in_progress"

    def _update_disease_statuses(self):
        for disease in self.diseases:
            if disease['status'] == 'cured':
                total_cubes_of_color = sum(city["cubes"][disease['color']] for city in self.board_state.values())
                if total_cubes_of_color == 0:
                    disease['status'] = 'eradicated'

    def _draw_card(self):
        if self.deck:
            self.player_hand.append(self.deck.pop(0))

    def _infect_city(self):
        if not self.infection_deck:
            random.shuffle(self.infection_discard)
            self.infection_deck = self.infection_discard
            self.infection_discard = []

        if not self.infection_deck: return

        city_name = self.infection_deck.pop(0)
        self.infection_discard.append(city_name)
        color = self.map[city_name]['color']

        if self._get_disease_status(color) == 'eradicated':
            return

        if self.board_state[city_name]["cubes"][color] < 3:
            self.board_state[city_name]["cubes"][color] += 1
        else:
            self._outbreak(city_name, color)

    def _outbreak(self, city, color):
        self.outbreaks += 1
        cities_in_outbreak = {city}
        q = deque([city])

        while q:
            current_city = q.popleft()
            for neighbor in self.map[current_city]["neighbors"]:
                if neighbor not in self.city_to_idx or self._get_disease_status(color) == 'eradicated':
                    continue

                if self.board_state[neighbor]["cubes"][color] < 3:
                    self.board_state[neighbor]["cubes"][color] += 1
                elif neighbor not in cities_in_outbreak:
                    cities_in_outbreak.add(neighbor)
                    q.append(neighbor)
                    self.outbreaks +=1

    def step(self, action_idx):
        action = self.idx_to_action[action_idx]
        action_type = action.get("type")

        if action_type == "move":
            self.player_location = self.idx_to_city[action["target_idx"]]
        elif action_type == "treat":
            city_to_treat = self.idx_to_city[action["target_idx"]]
            color_to_treat = action["color"]
            if self._is_disease_cured(color_to_treat):
                self.board_state[city_to_treat]["cubes"][color_to_treat] = 0
            else:
                if self.board_state[city_to_treat]["cubes"][color_to_treat] > 0:
                    self.board_state[city_to_treat]["cubes"][color_to_treat] -= 1
        elif action_type == "discover_cure":
            color_to_cure = action["color"]
            for disease in self.diseases:
                if disease['color'] == color_to_cure:
                    if disease['status'] == 'active':
                        disease['status'] = 'cured'
                        cards_of_color = [card for card in self.player_hand if self.map[card]['color'] == color_to_cure]
                        cards_to_discard = cards_of_color[:self.cards_for_cure]
                        for card in cards_to_discard:
                            self.player_hand.remove(card)
                    break
        elif action_type == "build_investigation_center":
            city_to_build_in = self.idx_to_city[action["target_idx"]]
            if city_to_build_in not in self.investigation_centers:
                self.investigation_centers.add(city_to_build_in)
                self.player_hand.remove(city_to_build_in)

        self.actions_taken += 1

        if self.actions_taken > 0 and self.actions_taken % 4 == 0:
            self._infect_city()

        self._update_disease_statuses()

        next_state = self.get_state_as_graph()
        done, result = self.is_game_over()
        reward = 0
        if done and result == "win":
            reward = 1000.0 / self.actions_taken if self.actions_taken > 0 else 1000.0
        elif done and result == "loss":
            reward = -500.0

        return next_state, reward, done

    def get_state_as_graph(self):
        node_features = []
        player_loc_idx = self.city_to_idx[self.player_location]
        hand_colors = Counter(self.map[card]['color'] for card in self.player_hand)

        disease_statuses = {d['color']: d['status'] for d in self.diseases}

        for i in range(len(self.all_cities)):
            city_name = self.idx_to_city[i]
            is_player = 1.0 if i == player_loc_idx else 0.0
            has_card = 1.0 if city_name in self.player_hand else 0.0
            has_center = 1.0 if city_name in self.investigation_centers else 0.0
            city_cubes = self.board_state[city_name]["cubes"]

            cube_features = [city_cubes[c] / 3.0 for c in self.all_possible_colors]
            cure_features = [1.0 if disease_statuses[c] in ['cured', 'eradicated'] else 0.0 for c in self.all_possible_colors]
            hand_features = [hand_colors.get(c, 0) / self.cards_for_cure for c in self.all_possible_colors]
            eradicated_features = [1.0 if disease_statuses[c] == 'eradicated' else 0.0 for c in self.all_possible_colors]

            features = cube_features + cure_features + hand_features + eradicated_features + [is_player, has_card, has_center]
            node_features.append(features)

        x = torch.tensor(node_features, dtype=torch.float)
        return Data(x=x, edge_index=self.edge_index)

    def get_node_feature_count(self):
        return len(self.all_possible_colors) * 4 + 3

    def _build_action_maps(self):
        self.action_to_idx = {}
        self.idx_to_action = {}
        action_idx_counter = 0
        for i in range(len(self.all_cities)):
            self.action_to_idx[json.dumps({"type": "move", "target_idx": i})] = action_idx_counter
            self.idx_to_action[action_idx_counter] = {"type": "move", "target_idx": i}; action_idx_counter += 1
        for i in range(len(self.all_cities)):
            for color in self.all_possible_colors:
                self.action_to_idx[json.dumps({"type": "treat", "target_idx": i, "color": color})] = action_idx_counter
                self.idx_to_action[action_idx_counter] = {"type": "treat", "target_idx": i, "color": color}; action_idx_counter += 1
        for color in self.all_possible_colors:
            self.action_to_idx[json.dumps({"type": "discover_cure", "color": color})] = action_idx_counter
            self.idx_to_action[action_idx_counter] = {"type": "discover_cure", "color": color}; action_idx_counter += 1
        for i in range(len(self.all_cities)):
            self.action_to_idx[json.dumps({"type": "build_investigation_center", "target_idx": i})] = action_idx_counter
            self.idx_to_action[action_idx_counter] = {"type": "build_investigation_center", "target_idx": i}; action_idx_counter += 1
        self.action_to_idx[json.dumps({"type": "pass"})] = action_idx_counter
        self.idx_to_action[action_idx_counter] = {"type": "pass"}

    def get_possible_action_mask(self):
        mask = [False] * len(self.action_to_idx)
        player_loc_idx = self.city_to_idx[self.player_location]

        for neighbor in self.map[self.player_location]["neighbors"]:
            if neighbor not in self.city_to_idx: continue
            neighbor_idx = self.city_to_idx[neighbor]
            mask[self.action_to_idx[json.dumps({"type": "move", "target_idx": neighbor_idx})]] = True

        for color in self.colors_in_play:
            if self.board_state[self.player_location]["cubes"][color] > 0:
                mask[self.action_to_idx[json.dumps({"type": "treat", "target_idx": player_loc_idx, "color": color})]] = True

        if self.player_location in self.investigation_centers:
            hand_colors = Counter(self.map[card]['color'] for card in self.player_hand)
            for disease in self.diseases:
                if disease['status'] == 'active' and hand_colors.get(disease['color'], 0) >= self.cards_for_cure:
                    mask[self.action_to_idx[json.dumps({"type": "discover_cure", "color": disease['color']})]] = True

        if self.player_location not in self.investigation_centers and self.player_location in self.player_hand:
            mask[self.action_to_idx[json.dumps({"type": "build_investigation_center", "target_idx": player_loc_idx})]] = True

        if not any(mask):
            mask[self.action_to_idx[json.dumps({"type": "pass"})]] = True

        return torch.tensor(mask, dtype=torch.bool)

    def _build_edge_index(self):
        edge_list = []
        for city, data in self.map.items():
            for neighbor in data["neighbors"]:
                if neighbor not in self.city_to_idx:
                    continue
                edge_list.append([self.city_to_idx[city], self.city_to_idx[neighbor]])
        self.edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()