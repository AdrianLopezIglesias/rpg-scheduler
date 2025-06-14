import random
import json
import os
from collections import deque
import torch
from torch_geometric.data import Data

class PandemicGame:
    def __init__(self, difficulty="easy", config=None):
        if not config:
            raise ValueError("A config object must be provided to initialize the game.")

        # The game now always uses the advanced ruleset.
        self.difficulty = difficulty
        self.colors = ["blue", "yellow", "black", "red"]

        self.map_config = self._load_map_config(difficulty)
        game_settings = config['game_settings'][difficulty]

        self.map = self.map_config["cities"]
        self.max_actions_per_game = game_settings["max_actions_per_game"]
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
        
        # Always use the more complex state, but handle legacy maps as special cases.
        self.board_state = {city: {"cubes": {color: 0 for color in self.colors}} for city in self.map}
        
        # For legacy difficulties, start with cures already discovered.
        if self.difficulty in ["tutorial", "easy", "hard"]:
            self.cures = {color: True for color in self.colors}
        else:
            self.cures = {color: False for color in self.colors}

        self._setup_initial_board()
        return self.get_state_as_graph()

    def _setup_initial_board(self):
        num_cubes_to_place = len(self.all_cities)
        for _ in range(num_cubes_to_place):
            city = random.choice(self.all_cities)
            city_color = self.map[city]['color']
            # Only place cubes of the city's color.
            if self.board_state[city]["cubes"][city_color] < 3:
                self.board_state[city]["cubes"][city_color] += 1

    def is_game_over(self):
        if self.actions_taken >= self.max_actions_per_game:
            return True, "loss"

        # If any cure is not yet discovered, the only way to win is to discover them all.
        if not all(self.cures.values()):
            return False, "in_progress"
        
        # If all cures ARE discovered (either at the start or by the player),
        # the win condition becomes clearing all cubes from the board.
        total_cubes = sum(sum(cubes.values()) for city, data in self.board_state.items() for cubes in [data['cubes']])
        if total_cubes == 0:
            return True, "win"

        return False, "in_progress"

    def step(self, action_idx):
        action = self.idx_to_action[action_idx]
        action_type = action.get("type")

        if action_type == "move":
            self.player_location = self.idx_to_city[action["target_idx"]]
        elif action_type == "treat":
            city_to_treat = self.idx_to_city[action["target_idx"]]
            color_to_treat = action["color"]
            if self.board_state[city_to_treat]["cubes"][color_to_treat] > 0:
                self.board_state[city_to_treat]["cubes"][color_to_treat] -= 1
        elif action_type == "discover_cure":
            color_to_cure = action["color"]
            if not self.cures[color_to_cure]:
                self.cures[color_to_cure] = True

        self.actions_taken += 1
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
        for i in range(len(self.all_cities)):
            city_name = self.idx_to_city[i]
            is_player = 1.0 if i == player_loc_idx else 0.0

            # Always use the more complex feature vector.
            city_cubes = self.board_state[city_name]["cubes"]
            cube_features = [city_cubes[c] / 3.0 for c in self.colors]
            cure_features = [1.0 if self.cures[c] else 0.0 for c in self.colors]
            features = cube_features + [is_player] + cure_features
            
            node_features.append(features)

        x = torch.tensor(node_features, dtype=torch.float)
        return Data(x=x, edge_index=self.edge_index)

    def get_node_feature_count(self):
        # The feature count is now always fixed to the advanced representation.
        return len(self.colors) + 1 + len(self.colors) # Cubes, Player, Cures

    def _build_action_maps(self):
        self.action_to_idx = {}
        self.idx_to_action = {}
        action_idx_counter = 0

        # Move actions
        for i in range(len(self.all_cities)):
            action = {"type": "move", "target_idx": i}
            self.action_to_idx[json.dumps(action)] = action_idx_counter
            self.idx_to_action[action_idx_counter] = action
            action_idx_counter += 1
        
        # Treat actions are now always by color.
        for i in range(len(self.all_cities)):
            for color in self.colors:
                action = {"type": "treat", "target_idx": i, "color": color}
                self.action_to_idx[json.dumps(action)] = action_idx_counter
                self.idx_to_action[action_idx_counter] = action
                action_idx_counter += 1

        # Discover Cure actions
        for color in self.colors:
            action = {"type": "discover_cure", "color": color}
            self.action_to_idx[json.dumps(action)] = action_idx_counter
            self.idx_to_action[action_idx_counter] = action
            action_idx_counter += 1

        # Pass action
        action = {"type": "pass"}
        self.action_to_idx[json.dumps(action)] = action_idx_counter
        self.idx_to_action[action_idx_counter] = action

    def get_possible_action_mask(self):
        mask = [False] * len(self.action_to_idx)
        player_loc_idx = self.city_to_idx[self.player_location]

        # Move
        for neighbor in self.map[self.player_location]["neighbors"]:
            neighbor_idx = self.city_to_idx[neighbor]
            action = {"type": "move", "target_idx": neighbor_idx}
            mask[self.action_to_idx[json.dumps(action)]] = True

        # Treat
        for color in self.colors:
            if self.board_state[self.player_location]["cubes"][color] > 0:
                action = {"type": "treat", "target_idx": player_loc_idx, "color": color}
                mask[self.action_to_idx[json.dumps(action)]] = True
        
        # Discover Cure
        for color in self.colors:
            if not self.cures[color]:
                action = {"type": "discover_cure", "color": color}
                mask[self.action_to_idx[json.dumps(action)]] = True

        # Pass
        if not any(mask):
            action = {"type": "pass"}
            mask[self.action_to_idx[json.dumps(action)]] = True

        return torch.tensor(mask, dtype=torch.bool)
        
    def _build_edge_index(self):
        edge_list = []
        for city, data in self.map.items():
            for neighbor in data["neighbors"]:
                edge_list.append([self.city_to_idx[city], self.city_to_idx[neighbor]])
        self.edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()