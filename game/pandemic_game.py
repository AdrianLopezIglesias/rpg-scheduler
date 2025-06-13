import random
import json
import os
from collections import deque

class PandemicGame:
    """
    The game environment, now loading its map and game settings from
    a passed config object.
    """
    def __init__(self, difficulty="easy", config=None):
        if not config:
            raise ValueError("A config object must be provided to initialize the game.")
            
        self.map_config = self._load_map_config(difficulty)
        game_settings = config['game_settings'][difficulty]

        self.map = self.map_config["cities"]
        self.max_actions_per_game = game_settings["max_actions_per_game"]
        self.all_cities = list(self.map.keys())
        self.distances = self._precompute_distances()
        self.reset()

    def _load_map_config(self, difficulty):
        """Loads the map data for the specified difficulty from the JSON file."""
        config_path = os.path.join(os.path.dirname(__file__), 'maps.json')
        with open(config_path, 'r') as f:
            return json.load(f)[difficulty]

    def _precompute_distances(self):
        """Calculates the shortest path between all cities using BFS."""
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
        """Resets the game to a new, random initial state."""
        self.board_state = {city: {"cubes": 0} for city in self.map}
        self.player_location = random.choice(self.all_cities)
        self.actions_taken = 0
        self._setup_initial_board()
        return self.get_state_snapshot()

    def _setup_initial_board(self):
        """Creates a new, random puzzle for the agent to solve."""
        num_cubes_to_place = random.randint(len(self.all_cities) * 2, len(self.all_cities) * 3)
        for _ in range(num_cubes_to_place):
            city = random.choice(self.all_cities)
            if self.board_state[city]["cubes"] < 3:
                self.board_state[city]["cubes"] += 1

    def get_state_snapshot(self):
        """Returns a serializable copy of the current game state."""
        return {
            "board": json.loads(json.dumps(self.board_state)),
            "player_location": self.player_location,
            "actions_taken": self.actions_taken
        }
    
    def get_possible_actions(self):
        """Returns a list of all valid single actions from the current state."""
        possible_actions = []
        if self.board_state[self.player_location]["cubes"] > 0:
            possible_actions.append({"type": "treat", "target": self.player_location})
        for neighbor in self.map[self.player_location]["neighbors"]:
            possible_actions.append({"type": "move", "target": neighbor})
        if not possible_actions:
            possible_actions.append({"type": "pass", "target": None})
        return possible_actions

    def is_game_over(self):
        """Checks for win or loss conditions."""
        total_cubes = sum(data["cubes"] for data in self.board_state.values())
        if total_cubes == 0:
            return "win"
        if self.actions_taken >= self.max_actions_per_game:
            return "loss"
        return False

    def step(self, action):
        """Processes a single player action and updates the game state."""
        if self.is_game_over():
            return self.get_state_snapshot()

        if action["type"] == "move":
            self.player_location = action["target"]
        elif action["type"] == "treat":
            if self.board_state[action["target"]]["cubes"] > 0:
                self.board_state[action["target"]]["cubes"] -= 1
        
        self.actions_taken += 1
        return self.get_state_snapshot()
