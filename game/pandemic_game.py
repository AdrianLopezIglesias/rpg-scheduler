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
        self.map_config = self._load_map_config(difficulty)
        game_settings = config['game_settings'][difficulty]
        self.map = self.map_config["cities"]
        self.max_actions_per_game = game_settings["max_actions_per_game"]
        self.all_cities = list(self.map.keys())
        self.distances = self._precompute_distances()
        self.city_to_idx = {city: i for i, city in enumerate(self.all_cities)}
        self.idx_to_city = {i: city for city, i in self.city_to_idx.items()}
        self._build_action_maps()
        self._build_edge_index()
        self.reset()

    def _load_map_config(self, difficulty):
        config_path = os.path.join(os.path.dirname(__file__), 'maps.json')
        with open(config_path, 'r') as f:
            return json.load(f)[difficulty]

    def _precompute_distances(self):
        distances = {}
        for start_node in self.all_cities:
            distances[start_node] = {node: float('inf') for node in self.all_cities}
            distances[start_node][start_node] = 0
            queue = deque([start_node])
            visited = {start_node}
            while queue:
                current_node = queue.popleft()
                for neighbor in self.map[current_node]["neighbors"]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[start_node][neighbor] = distances[start_node][current_node] + 1
                        queue.append(neighbor)
        return distances

    def get_distance(self, city1, city2):
        return self.distances[city1].get(city2, float('inf'))

    def reset(self):
        self.board_state = {city: {"cubes": 0} for city in self.map}
        self.player_location = random.choice(self.all_cities)
        self.actions_taken = 0
        self._setup_initial_board()
        return self.get_state_as_graph()

    def _setup_initial_board(self):
        # Increased the number of cubes for a harder challenge
        num_cubes_to_place = len(self.all_cities)
        for _ in range(num_cubes_to_place):
            city = random.choice(self.all_cities)
            if self.board_state[city]["cubes"] < 3:
                self.board_state[city]["cubes"] += 1

    def get_state_snapshot(self):
        return {
            "board": json.loads(json.dumps(self.board_state)),
            "player_location": self.player_location,
            "actions_taken": self.actions_taken
        }

    def get_possible_actions(self):
        possible_actions = []
        if self.board_state[self.player_location]["cubes"] > 0:
            possible_actions.append({"type": "treat", "target": self.player_location})
        for neighbor in self.map[self.player_location]["neighbors"]:
            possible_actions.append({"type": "move", "target": neighbor})
        if not possible_actions:
            possible_actions.append({"type": "pass", "target": None})
        return possible_actions

    def is_game_over(self):
        total_cubes = sum(data["cubes"] for data in self.board_state.values())
        if total_cubes == 0:
            return True, "win"
        if self.actions_taken >= self.max_actions_per_game:
            return True, "loss"
        return False, "in_progress"

    def step(self, action_idx):
        action = self.idx_to_action[action_idx]
        if action["type"] == "move":
            self.player_location = self.idx_to_city[action["target_idx"]]
        elif action["type"] == "treat":
            city_to_treat = self.idx_to_city[action["target_idx"]]
            if self.board_state[city_to_treat]["cubes"] > 0:
                self.board_state[city_to_treat]["cubes"] -= 1
        self.actions_taken += 1
        next_state = self.get_state_as_graph()
        done, result = self.is_game_over()
        reward = 0
        if done:
            if result == "win":
                reward = 1000.0 / self.actions_taken if self.actions_taken > 0 else 1000.0
            else:
                reward = -500.0
        return next_state, reward, done

    def _build_action_maps(self):
        self.action_to_idx = {}
        self.idx_to_action = {}
        action_idx_counter = 0
        for i in range(len(self.all_cities)):
            action = {"type": "treat", "target_idx": i}
            self.action_to_idx[json.dumps(action)] = action_idx_counter
            self.idx_to_action[action_idx_counter] = action
            action_idx_counter += 1
        for i in range(len(self.all_cities)):
            action = {"type": "move", "target_idx": i}
            self.action_to_idx[json.dumps(action)] = action_idx_counter
            self.idx_to_action[action_idx_counter] = action
            action_idx_counter += 1
        action = {"type": "pass", "target_idx": -1}
        self.action_to_idx[json.dumps(action)] = action_idx_counter
        self.idx_to_action[action_idx_counter] = action

    def _build_edge_index(self):
        edge_list = []
        for city, data in self.map.items():
            for neighbor in data["neighbors"]:
                edge_list.append([self.city_to_idx[city], self.city_to_idx[neighbor]])
        self.edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    def get_state_as_graph(self):
        node_features = []
        player_loc_idx = self.city_to_idx[self.player_location]
        for i in range(len(self.all_cities)):
            city_name = self.idx_to_city[i]
            cubes = self.board_state[city_name]["cubes"] / 3.0
            is_player = 1.0 if i == player_loc_idx else 0.0
            node_features.append([cubes, is_player])
        x = torch.tensor(node_features, dtype=torch.float)
        return Data(x=x, edge_index=self.edge_index)

    def get_possible_action_mask(self):
        mask = [False] * self.get_action_space_size()
        player_loc_idx = self.city_to_idx[self.player_location]
        if self.board_state[self.player_location]["cubes"] > 0:
            action = {"type": "treat", "target_idx": player_loc_idx}
            mask[self.action_to_idx[json.dumps(action)]] = True
        for neighbor in self.map[self.player_location]["neighbors"]:
            neighbor_idx = self.city_to_idx[neighbor]
            action = {"type": "move", "target_idx": neighbor_idx}
            mask[self.action_to_idx[json.dumps(action)]] = True
        if not any(mask):
            action = {"type": "pass", "target_idx": -1}
            mask[self.action_to_idx[json.dumps(action)]] = True
        return torch.tensor(mask, dtype=torch.bool)

    def get_action_space_size(self):
        return len(self.idx_to_action)
