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
        
        self.deck = [city for city in self.all_cities]
        random.shuffle(self.deck)
        self.player_hand = []
        
        self.board_state = {city: {"cubes": {color: 0 for color in self.all_possible_colors}} for city in self.map}
        
        # Set up cures based on the new `cures_found` config.
        self.cures = {color: False for color in self.all_possible_colors}
        cures_already_found = self.map_config.get("cures_found", [])
        for color in cures_already_found:
            if color in self.cures:
                self.cures[color] = True

        self._setup_initial_board()
        for _ in range(3): self._draw_card()
        return self.get_state_as_graph()

    def _setup_initial_board(self):
        # Read the cube setup from the map configuration
        cube_config = self.map_config.get("initial_cubes", {})
        
        # Create a shuffled list of cities to draw from without replacement
        cities_to_infect = list(self.all_cities)
        random.shuffle(cities_to_infect)

        # Helper function to place cubes
        def place_cubes(num_cities, num_cubes):
            for _ in range(num_cities):
                if not cities_to_infect:
                    # Stop if we run out of cities
                    break
                city_name = cities_to_infect.pop(0)
                city_color = self.map[city_name]['color']
                # Only place cubes if the color is active for this difficulty
                if city_color in self.colors_in_play:
                    self.board_state[city_name]["cubes"][city_color] = num_cubes
        
        # Place cubes according to the configuration
        place_cubes(cube_config.get("three_cubes", 0), 3)
        place_cubes(cube_config.get("two_cubes", 0), 2)
        place_cubes(cube_config.get("one_cube", 0), 1)

    def is_game_over(self):
        if self.actions_taken >= self.max_actions_per_game:
            return True, "loss"

        total_cubes = sum(sum(cubes.values()) for cubes in [data['cubes'] for data in self.board_state.values()])
        if total_cubes == 0:
            return True, "win"

        return False, "in_progress"
    
    def _draw_card(self):
        if self.deck:
            card = self.deck.pop(0)
            self.player_hand.append(card)

    def step(self, action_idx):
        action = self.idx_to_action[action_idx]
        action_type = action.get("type")

        if action_type == "move":
            self.player_location = self.idx_to_city[action["target_idx"]]
        elif action_type == "treat":
            city_to_treat = self.idx_to_city[action["target_idx"]]
            color_to_treat = action["color"]
            if self.cures[color_to_treat]:
                self.board_state[city_to_treat]["cubes"][color_to_treat] = 0
            else:
                if self.board_state[city_to_treat]["cubes"][color_to_treat] > 0:
                    self.board_state[city_to_treat]["cubes"][color_to_treat] -= 1
        elif action_type == "discover_cure":
            color_to_cure = action["color"]
            if not self.cures[color_to_cure]:
                self.cures[color_to_cure] = True
                cards_of_color = [card for card in self.player_hand if self.map[card]['color'] == color_to_cure]
                cards_to_discard = cards_of_color[:self.cards_for_cure]
                for card in cards_to_discard:
                    self.player_hand.remove(card)

        self.actions_taken += 1
        
        if self.actions_taken > 0 and self.actions_taken % 4 == 0:
            self._draw_card()
        
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
            city_cubes = self.board_state[city_name]["cubes"]
            
            cube_features = [city_cubes[c] / 3.0 for c in self.all_possible_colors]
            cure_features = [1.0 if self.cures[c] else 0.0 for c in self.all_possible_colors]
            hand_features = [hand_colors.get(c, 0) / self.cards_for_cure for c in self.all_possible_colors]
            
            features = cube_features + cure_features + hand_features + [is_player, has_card]
            node_features.append(features)

        x = torch.tensor(node_features, dtype=torch.float)
        return Data(x=x, edge_index=self.edge_index)

    def get_node_feature_count(self):
        return len(self.all_possible_colors) * 3 + 2

    def _build_action_maps(self):
        self.action_to_idx = {}
        self.idx_to_action = {}
        action_idx_counter = 0
        for i in range(len(self.all_cities)):
            self.action_to_idx[json.dumps({"type": "move", "target_idx": i})] = action_idx_counter
            self.idx_to_action[action_idx_counter] = {"type": "move", "target_idx": i}
            action_idx_counter += 1
        for i in range(len(self.all_cities)):
            for color in self.all_possible_colors:
                self.action_to_idx[json.dumps({"type": "treat", "target_idx": i, "color": color})] = action_idx_counter
                self.idx_to_action[action_idx_counter] = {"type": "treat", "target_idx": i, "color": color}
                action_idx_counter += 1
        for color in self.all_possible_colors:
            self.action_to_idx[json.dumps({"type": "discover_cure", "color": color})] = action_idx_counter
            self.idx_to_action[action_idx_counter] = {"type": "discover_cure", "color": color}
            action_idx_counter += 1
        self.action_to_idx[json.dumps({"type": "pass"})] = action_idx_counter
        self.idx_to_action[action_idx_counter] = {"type": "pass"}

    def get_possible_action_mask(self):
        mask = [False] * len(self.action_to_idx)
        player_loc_idx = self.city_to_idx[self.player_location]

        for neighbor in self.map[self.player_location]["neighbors"]:
            neighbor_idx = self.city_to_idx[neighbor]
            mask[self.action_to_idx[json.dumps({"type": "move", "target_idx": neighbor_idx})]] = True

        for color in self.colors_in_play:
            if self.board_state[self.player_location]["cubes"][color] > 0:
                mask[self.action_to_idx[json.dumps({"type": "treat", "target_idx": player_loc_idx, "color": color})]] = True
        
        hand_colors = Counter(self.map[card]['color'] for card in self.player_hand)
        for color in self.colors_in_play:
            if not self.cures[color] and hand_colors.get(color, 0) >= self.cards_for_cure:
                mask[self.action_to_idx[json.dumps({"type": "discover_cure", "color": color})]] = True

        if not any(mask):
            mask[self.action_to_idx[json.dumps({"type": "pass"})]] = True

        return torch.tensor(mask, dtype=torch.bool)
        
    def _build_edge_index(self):
        edge_list = []
        for city, data in self.map.items():
            for neighbor in data["neighbors"]:
                edge_list.append([self.city_to_idx[city], self.city_to_idx[neighbor]])
        self.edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()