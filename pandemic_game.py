import random
import json

class PandemicGame:
    """
    Encapsulates the rules and state of a simplified Pandemic game.
    This class is completely independent of any player or AI agent.
    """
    def __init__(self):
        self.map = self._create_map()
        self.max_cubes_per_city = 3
        self.outbreak_limit = 1
        self.max_rounds = 10
        self.actions_per_turn = 4
        self.reset()

    def _create_map(self):
        # This remains the same simplified map
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
            "Tokyo": {"color": "red", "neighbors": ["San Francisco"]}, "Manila": {"color": "red", "neighbors": ["San Francisco"]},
            "London": {"color": "blue", "neighbors": ["New York"]}, "Madrid": {"color": "blue", "neighbors": ["New York"]},
            "Sydney": {"color": "red", "neighbors": ["Los Angeles"]}, "Lima": {"color": "yellow", "neighbors": ["Mexico City"]},
            "Bogota": {"color": "yellow", "neighbors": ["Mexico City", "Miami"]},
        }

    def reset(self):
        """Resets the game to a clean initial state."""
        self.board_state = {city: {"cubes": 0} for city in self.map}
        self.player_location = "Atlanta"
        self.outbreaks = 0
        self.current_round = 1
        self.actions_remaining = self.actions_per_turn
        self.fitness_score = 0
        self._initial_infection()
        return self.get_state_snapshot()

    def _initial_infection(self):
        blue_cities = [city for city, data in self.map.items() if data["color"] == "blue"]
        infected_cities = random.sample(blue_cities, 3)
        for city in infected_cities:
            self.board_state[city]["cubes"] = 2

    def get_state_snapshot(self):
        """Returns a serializable copy of the current game state."""
        return {
            "board": json.loads(json.dumps(self.board_state)),
            "player_location": self.player_location,
            "outbreaks": self.outbreaks,
            "current_round": self.current_round,
            "actions_remaining": self.actions_remaining
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
        """Checks if the game has reached a win or loss condition."""
        if self.outbreaks >= self.outbreak_limit:
            return "loss"
        if self.current_round > self.max_rounds:
            return "win"
        return False

    def step(self, action):
        """
        Processes a single player action and updates the game state.
        Handles the infection phase automatically after 4 actions.
        """
        # 1. Execute player action
        if action["type"] == "move":
            self.player_location = action["target"]
        elif action["type"] == "treat":
            if self.board_state[action["target"]]["cubes"] == 3:
                self.fitness_score += 50
            else:
                self.fitness_score += 10
            self.board_state[action["target"]]["cubes"] -= 1

        self.actions_remaining -= 1

        # 2. If turn is over, handle infection phase
        if self.actions_remaining <= 0:
            self.current_round += 1
            blue_cities = [c for c, data in self.map.items() if data["color"] == "blue"]
            city_to_infect = random.choice(blue_cities)
            
            # Infect and check for outbreak
            if self.board_state[city_to_infect]["cubes"] + 1 > self.max_cubes_per_city:
                self.outbreaks += 1
                self.fitness_score -= 100
            else:
                self.board_state[city_to_infect]["cubes"] += 1

            self.actions_remaining = self.actions_per_turn # Reset for next turn

        return self.get_state_snapshot()

