import random
import json

class PandemicGame:
    """
    A redesigned, deterministic game environment with a smaller map and more turns
    to facilitate AI learning.
    """
    def __init__(self):
        self.map = self._create_map()
        self.max_actions_per_game = 1000 # Increased from 100
        self.all_cities = list(self.map.keys())
        self.reset()

    def _create_map(self):
        """A smaller, more manageable map for the agent to learn on."""
        return {
            "Atlanta": {"neighbors": ["Chicago", "Washington"]},
            "Chicago": {"neighbors": ["Atlanta", "Montreal"]},
            "Washington": {"neighbors": ["Atlanta", "New York"]},
            "Montreal": {"neighbors": ["Chicago", "New York"]},
            "New York": {"neighbors": ["Washington", "Montreal"]},
        }

    def reset(self):
        """Resets the game to a new, random initial state."""
        self.board_state = {city: {"cubes": 0} for city in self.map}
        self.player_location = "Atlanta"
        self.actions_taken = 0
        self._setup_initial_board()
        return self.get_state_snapshot()

    def _setup_initial_board(self):
        """Creates a new, random puzzle on the smaller board."""
        # Increased number of cubes to create a more challenging start.
        num_cubes_to_place = random.randint(10, 15)
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
