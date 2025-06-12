import random
import json

class PandemicGame:
    """
    The game environment, now with a larger, more complex map and a more
    challenging initial setup to test the agent's strategic capabilities.
    """
    def __init__(self):
        self.map = self._create_map()
        self.max_actions_per_game = 250
        self.all_cities = list(self.map.keys())
        self.reset()

    def _create_map(self):
        """The full-size map, restored to increase game complexity."""
        return {
            "San Francisco": {"color": "blue", "neighbors": ["Chicago", "Los Angeles", "Tokyo", "Manila"]},
            "Chicago": {"color": "blue", "neighbors": ["San Francisco", "Los Angeles", "Mexico City", "Atlanta", "Montreal"]},
            "Atlanta": {"color": "blue", "neighbors": ["Chicago", "Washington", "Miami"]},
            "Montreal": {"color": "blue", "neighbors": ["Chicago", "Washington", "New York"]},
            "Washington": {"color": "blue", "neighbors": ["Atlanta", "Montreal", "New York", "Miami"]},
            "New York": {"color": "blue", "neighbors": ["Montreal", "Washington", "London", "Madrid"]},
            "Los Angeles": {"color": "blue", "neighbors": ["San Francisco", "Chicago", "Mexico City", "Sydney"]},
            "Mexico City": {"color": "blue", "neighbors": ["Los Angeles", "Chicago", "Miami", "Lima", "Bogota"]},
            "Miami": {"color": "blue", "neighbors": ["Atlanta", "Mexico City", "Washington", "Bogota"]},
            "Tokyo": {"color": "red", "neighbors": ["San Francisco"]},
            "Manila": {"color": "red", "neighbors": ["San Francisco"]},
            "London": {"color": "blue", "neighbors": ["New York"]},
            "Madrid": {"color": "blue", "neighbors": ["New York"]},
            "Sydney": {"color": "red", "neighbors": ["Los Angeles"]},
            "Lima": {"color": "yellow", "neighbors": ["Mexico City"]},
            "Bogota": {"color": "yellow", "neighbors": ["Mexico City", "Miami"]},
        }

    def reset(self):
        """Resets the game to a new, random initial state."""
        self.board_state = {city: {"cubes": 0} for city in self.map}
        self.player_location = "Atlanta"
        self.actions_taken = 0
        self._setup_initial_board()
        return self.get_state_snapshot()

    def _setup_initial_board(self):
        """Creates a more difficult starting board state."""
        # Increased number of cubes to create a more challenging start.
        num_cubes_to_place = random.randint(25, 40)
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
