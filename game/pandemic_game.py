import random
import json
import os
from collections import deque, Counter
import torch
from torch_geometric.data import Data

class PandemicGame:
    def __init__(self, difficulty="easy", config=None):
        if not config:
            raise ValueError("A config object must be provided to initialize the game.")

        self.difficulty = difficulty
        self.all_possible_colors = ["blue", "yellow", "black", "red"]
        self.map_config = self._load_map_config(difficulty)
        self.colors_in_play = self.map_config["colors_in_play"]
        game_settings = config['game_settings']
        self.cards_for_cure = game_settings['cards_for_cure']
        self.map = self.map_config["cities"]
        self.max_actions_per_game = game_settings[difficulty]["max_actions_per_game"]
        self.all_cities = list(self.map.keys())
        self.city_to_idx = {city: i for i, city in enumerate(self.all_cities)}
        self.idx_to_city = {i: city for city, i in self.city_to_idx.items()}
        self._build_action_maps()
        self._build_edge_index()
        self.reset()

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
        self.cures = {color: {"found": False, "eradicated": False} for color in self.all_possible_colors}
        cures_already_found = self.map_config.get("cures_found", [])
        for color in cures_already_found:
            if color in self.cures:
                self.cures[color]["found"] = True
        
        self.infection_deck = [city for city, data in self.map.items() if data['color'] in self.colors_in_play]
        random.shuffle(self.infection_deck)
        self.infection_discard = []

        self._setup_initial_board()
        for _ in range(3): self._draw_card()
        return self.get_state_as_graph()

    def _setup_initial_board(self):
        # New setup logic: draw 9 cards and place cubes accordingly.
        for i in range(3):
            self._infect_city(3)
        for i in range(3):
            self._infect_city(2)
        for i in range(3):
            self._infect_city(1)

    def is_game_over(self):
        if self.actions_taken >= self.max_actions_per_game or self.outbreaks >= 8:
            return True, "loss"

        # Win condition: all diseases in play must be eradicated.
        diseases_eradicated = 0
        for color in self.colors_in_play:
            if self.cures[color]["eradicated"]:
                diseases_eradicated += 1
        
        if diseases_eradicated == len(self.colors_in_play):
            return True, "win"

        return False, "in_progress"

    def _update_eradication_status(self):
        for color in self.colors_in_play:
            if self.cures[color]["found"] and not self.cures[color]["eradicated"]:
                total_cubes_of_color = sum(city["cubes"][color] for city in self.board_state.values())
                if total_cubes_of_color == 0:
                    self.cures[color]["eradicated"] = True
    
    def _draw_card(self):
        if self.deck:
            card = self.deck.pop(0)
            self.player_hand.append(card)

    def _infect_city(self, num_cubes=1):
        if not self.infection_deck:
            random.shuffle(self.infection_discard)
            self.infection_deck = self.infection_discard
            self.infection_discard = []
        
        city_name = self.infection_deck.pop(0)
        self.infection_discard.append(city_name)
        color = self.map[city_name]['color']
        
        if self.cures[color]["eradicated"]:
            return

        for _ in range(num_cubes):
            if self.board_state[city_name]["cubes"][color] < 3:
                self.board_state[city_name]["cubes"][color] += 1
            else:
                self._outbreak(city_name, color)
                # An outbreak can trigger a chain reaction, but we stop after the first.
                break

    def _outbreak(self, city, color):
        self.outbreaks += 1
        cities_in_outbreak = {city}
        q = deque([city])
        
        while q:
            current_city = q.popleft()
            for neighbor in self.map[current_city]["neighbors"]:
                if neighbor not in self.city_to_idx or self.cures[color]["eradicated"]:
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
            if self.cures[color_to_treat]["found"]:
                self.board_state[city_to_treat]["cubes"][color_to_treat] = 0
            else:
                if self.board_state[city_to_treat]["cubes"][color_to_treat] > 0:
                    self.board_state[city_to_treat]["cubes"][color_to_treat] -= 1
        elif action_type == "discover_cure":
            color_to_cure = action["color"]
            if not self.cures[color_to_cure]["found"]:
                self.cures[color_to_cure]["found"] = True
                cards_of_color = [card for card in self.player_hand if self.map[card]['color'] == color_to_cure]
                cards_to_discard = cards_of_color[:self.cards_for_cure]
                for card in cards_to_discard:
                    self.player_hand.remove(card)
        elif action_type == "build_investigation_center":
            city_to_build_in = self.idx_to_city[action["target_idx"]]
            if city_to_build_in not in self.investigation_centers:
                self.investigation_centers.add(city_to_build_in)
                self.player_hand.remove(city_to_build_in)

        self.actions_taken += 1
        
        # Infection Phase
        if self.actions_taken > 0 and self.actions_taken % 4 == 0:
            self._infect_city()

        self._update_eradication_status()
        
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

        for i in range(len(self.all_cities)):
            city_name = self.idx_to_city[i]
            is_player = 1.0 if i == player_loc_idx else 0.0
            has_card = 1.0 if city_name in self.player_hand else 0.0
            has_center = 1.0 if city_name in self.investigation_centers else 0.0
            city_cubes = self.board_state[city_name]["cubes"]
            
            cube_features = [city_cubes[c] / 3.0 for c in self.all_possible_colors]
            cure_features = [1.0 if self.cures[c]["found"] else 0.0 for c in self.all_possible_colors]
            hand_features = [hand_colors.get(c, 0) / self.cards_for_cure for c in self.all_possible_colors]
            eradicated_features = [1.0 if self.cures[c]["eradicated"] else 0.0 for c in self.all_possible_colors]

            features = cube_features + cure_features + hand_features + eradicated_features + [is_player, has_card, has_center]
            node_features.append(features)

        x = torch.tensor(node_features, dtype=torch.float)
        return Data(x=x, edge_index=self.edge_index)

    def get_node_feature_count(self):
        return len(self.all_possible_colors) * 4 + 3

    def _build_action_maps(self):
        # This method is unchanged
        pass

    def get_possible_action_mask(self):
        # This method is unchanged, but the logic inside `discover_cure` now checks `cures[color]['found']`
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
            for color in self.colors_in_play:
                if not self.cures[color]["found"] and hand_colors.get(color, 0) >= self.cards_for_cure:
                    mask[self.action_to_idx[json.dumps({"type": "discover_cure", "color": color})]] = True
        
        if self.player_location not in self.investigation_centers and self.player_location in self.player_hand:
            mask[self.action_to_idx[json.dumps({"type": "build_investigation_center", "target_idx": player_loc_idx})]] = True

        if not any(mask):
            mask[self.action_to_idx[json.dumps({"type": "pass"})]] = True

        return torch.tensor(mask, dtype=torch.bool)
        
    def _build_edge_index(self):
        # This method is unchanged
        pass