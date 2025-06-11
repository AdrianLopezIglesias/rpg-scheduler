import random
import json

class PandemicSimulator:
    """
    A simplified simulation of the Pandemic board game, updated to generate
    turn-by-turn data with a fitness score for AI training.
    """

    def __init__(self):
        """Initializes the game with a fixed map and game state."""
        self.map = self._create_map()
        self.max_cubes_per_city = 3
        self.outbreak_limit = 1 # A single outbreak causes a loss
        self.max_rounds = 10
        self.reset()

    def _create_map(self):
        """Creates the game board focused on the 'blue' region."""
        # A simplified graph of North American and connected cities
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
            # Connections to other regions
            "Tokyo": {"color": "red", "neighbors": ["San Francisco"]},
            "Manila": {"color": "red", "neighbors": ["San Francisco"]},
            "London": {"color": "blue", "neighbors": ["New York"]},
            "Madrid": {"color": "blue", "neighbors": ["New York"]},
            "Sydney": {"color": "red", "neighbors": ["Los Angeles"]},
            "Lima": {"color": "yellow", "neighbors": ["Mexico City"]},
            "Bogota": {"color": "yellow", "neighbors": ["Mexico City", "Miami"]},
        }

    def reset(self):
        """Resets the game state to its initial condition for a new simulation."""
        self.board_state = {}
        for city in self.map:
            self.board_state[city] = {"cubes": 0}
        
        self.player_location = "Atlanta"
        self.outbreaks = 0
        self.current_round = 1
        self.game_over = False
        self.game_result = None
        self.fitness_score = 0  # Initialize fitness score

        self._initial_infection()

    def _initial_infection(self):
        """Places the starting disease cubes on the board."""
        blue_cities = [city for city, data in self.map.items() if data["color"] == "blue"]
        infected_cities = random.sample(blue_cities, 3)
        for city in infected_cities:
            # Using the infect method directly is not ideal for initial setup as it could cause an instant loss.
            # For simplicity here, we assume initial infection won't cause an outbreak.
            self.board_state[city]["cubes"] = 2

    def get_state_snapshot(self):
        """Returns a copy of the current game state."""
        return {
            "board": json.loads(json.dumps(self.board_state)),
            "player_location": self.player_location,
            "outbreaks": self.outbreaks,
            "current_round": self.current_round
        }

    def infect(self, city, num_cubes=1):
        """Adds disease cubes to a city and handles outbreaks, updating fitness score."""
        if self.game_over:
            return

        if self.board_state[city]["cubes"] + num_cubes > self.max_cubes_per_city:
            self.outbreaks += 1
            self.fitness_score -= 100  # Penalty for losing
        else:
            self.board_state[city]["cubes"] += num_cubes

    def treat(self, city):
        """Removes one disease cube, updating fitness score based on the situation."""
        if self.board_state[city]["cubes"] > 0:
            if self.board_state[city]["cubes"] == 3:
                self.fitness_score += 50  # Bonus for controlling an outbreak threat
            else:
                self.fitness_score += 10  # Standard points for removing a cube
            self.board_state[city]["cubes"] -= 1
            return True
        return False

    def move(self, destination_city):
        """Moves the player to an adjacent city."""
        if destination_city in self.map[self.player_location]["neighbors"]:
            self.player_location = destination_city
            return True
        return False

    def get_possible_actions(self):
        """Returns a list of all valid actions the player can take."""
        possible_actions = []
        if self.board_state[self.player_location]["cubes"] > 0:
            possible_actions.append({"type": "treat", "target": self.player_location})
        for neighbor in self.map[self.player_location]["neighbors"]:
            possible_actions.append({"type": "move", "target": neighbor})
        if not possible_actions:
            possible_actions.append({"type": "pass", "target": None})
        return possible_actions

    def choose_random_action(self):
        """Selects a random valid action."""
        return random.choice(self.get_possible_actions())

    def step(self, action_sequence):
        """Processes one full round and returns the city that was infected."""
        if self.game_over:
            return None, self.game_result

        for action in action_sequence:
            if action["type"] == "move":
                self.move(action["target"])
            elif action["type"] == "treat":
                self.treat(action["target"])
        
        blue_cities = [city for city, data in self.map.items() if data["color"] == "blue"]
        city_to_infect = random.choice(blue_cities)
        self.infect(city_to_infect, 1)

        if self.outbreaks >= self.outbreak_limit:
            self.game_over = True
            self.game_result = "loss"
        elif self.current_round >= self.max_rounds:
            self.game_over = True
            self.game_result = "win"
        
        self.current_round += 1
        return city_to_infect, self.game_result

    def run_random_game(self):
        """
        Simulates a full game and returns its history in the specified format.
        """
        self.reset()
        turn_history = []

        while not self.game_over:
            initial_state = self.get_state_snapshot()
            actions = [self.choose_random_action() for _ in range(4)]
            
            newly_infected_city, _ = self.step(actions)

            turn_history.append({
                'initialState': initial_state,
                'playerActions': actions,
                'newInfections': [{'city': newly_infected_city, 'color': 'blue'}] if newly_infected_city else [],
                'fitnessScore': self.fitness_score,
                'end_result': self.game_result if self.game_over else "in_progress"
            })
            
        return turn_history

# --- Main execution block to generate and save simulation data ---
if __name__ == "__main__":
    simulator = PandemicSimulator()
    num_simulations = 1000
    all_turns_data = []
    
    print(f"--- Running {num_simulations} simulations to generate data ---")
    
    for i in range(num_simulations):
        game_turns = simulator.run_random_game()
        all_turns_data.extend(game_turns)
        if (i + 1) % 100 == 0:
            print(f"  ...completed {i + 1}/{num_simulations} simulations.")
            
    output_filename = "simulation_data_fitness.json"
    with open(output_filename, "w") as f:
        json.dump(all_turns_data, f, indent=2)
        
    print(f"\nSuccessfully generated and saved data to {output_filename}.")
    print(f"Total turns recorded: {len(all_turns_data)}")
