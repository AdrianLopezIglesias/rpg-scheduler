# simulation/simulator.py

import math
from simulation.game_state import GameState
from simulation.actions import Action, MoveAction, HoldAction

class Simulator:
    """
    Manages the simulation logic, updating the game state based on agent actions.
    """
    def __init__(self, config):
        self.config = config
        self.width = config['world_size']['width']
        self.height = config['world_size']['height']
        self.move_dist = config['agent']['move_distance']
        
        # Start the agent in the center
        initial_x = self.width // 2
        initial_y = self.height // 2
        self.state = GameState(agent_x=initial_x, agent_y=initial_y)

    def _is_within_bounds(self, x, y):
        """Checks if the given coordinates are within the world boundaries."""
        return 0 <= x < self.width and 0 <= y < self.height

    def step(self, action: Action):
        """
        Processes one action and updates the simulation state.
        """
        if isinstance(action, HoldAction):
            # If action is to hold, do nothing.
            print("Agent holds.")
            return

        if isinstance(action, MoveAction):
            # Calculate potential new position
            angle_rad = math.radians(action.angle)
            delta_x = self.move_dist * math.cos(angle_rad)
            delta_y = self.move_dist * math.sin(angle_rad)

            new_x = self.state.agent_x + delta_x
            new_y = self.state.agent_y + delta_y

            # Check for collisions with walls
            if self._is_within_bounds(new_x, new_y):
                self.state.agent_x = int(new_x)
                self.state.agent_y = int(new_y)
                print(f"Agent moves to ({self.state.agent_x}, {self.state.agent_y})")
            else:
                print("Move blocked by wall. Position unchanged.")
