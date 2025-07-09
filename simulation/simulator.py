# simulation/simulator.py

import math
import random
from simulation.game_state import GameState, Threat
from simulation.actions import Action, MoveAction, HoldAction, AttackAction

class Simulator:
    """
    Manages the simulation logic, updating the game state based on agent and threat actions.
    """
    def __init__(self, config):
        self.config = config
        self.width = config['world_size']['width']
        self.height = config['world_size']['height']
        self.agent_move_dist = config['agent']['move_distance']
        
        # Initialize agent state
        initial_agent_x = self.width // 2
        initial_agent_y = self.height // 2
        agent_strength = config['agent']['strength']

        # Initialize threats
        initial_threats = []
        threat_config = config['threats']['basic_random']
        for i in range(threat_config['count']):
            threat = Threat(
                id=i,
                x=800, 
                y=800,
                strength=threat_config['strength']
            )
            initial_threats.append(threat)

        self.state = GameState(
            agent_x=initial_agent_x, 
            agent_y=initial_agent_y,
            agent_strength=agent_strength,
            threats=initial_threats
        )

    def _is_within_bounds(self, x, y):
        """Checks if the given coordinates are within the world boundaries."""
        return 0 <= x < self.width and 0 <= y < self.height

    def _get_distance(self, x1, y1, x2, y2):
        """Calculates the Euclidean distance between two points."""
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def _process_threat_turns(self):
        """Processes the actions for all threats. They attack if in range, otherwise they move."""
        threat_config = self.config['threats']['basic_random']
        threat_move_dist = threat_config['move_distance']

        # Use a copy of the list to avoid issues if a threat is removed
        for threat in list(self.state.threats):
            # Check if the player is within the threat's movement/attack range.
            distance_to_agent = self._get_distance(threat.x, threat.y, self.state.agent_x, self.state.agent_y)
            
            if distance_to_agent <= threat_move_dist:
                # If in range, the threat always chooses to attack.
                # The outcome of the attack depends on strength.
                if threat.strength > self.state.agent_strength:
                    print(f"Threat {threat.id} attacks and defeats the player! GAME OVER.")
                    self.state.game_over = True
                    return # End turn processing immediately
                else:
                    print(f"Threat {threat.id} attacks but is too weak. The attack fails.")
                    # The threat's turn ends; it doesn't move after a failed attack.
            else:
                # If the player is not in range, move randomly.
                random_angle = random.uniform(0, 360)
                angle_rad = math.radians(random_angle)
                delta_x = threat_move_dist * math.cos(angle_rad)
                delta_y = threat_move_dist * math.sin(angle_rad)

                new_tx = threat.x + delta_x
                new_ty = threat.y + delta_y

                if self._is_within_bounds(new_tx, new_ty):
                    threat.x, threat.y = int(new_tx), int(new_ty)
                    print(f"Threat {threat.id} moves to ({threat.x}, {threat.y})")
                else:
                    print(f"Threat {threat.id} move blocked by wall. Position unchanged.")

    def step(self, action: Action):
        """
        Processes one player action, updates the simulation state, and then processes threat turns.
        """
        if self.state.game_over:
            print("Game is over. No more actions can be taken.")
            return

        player_turn_ended = False

        if isinstance(action, HoldAction):
            print("Agent holds.")
            player_turn_ended = True
        
        elif isinstance(action, MoveAction):
            angle_rad = math.radians(action.angle)
            delta_x = self.agent_move_dist * math.cos(angle_rad)
            delta_y = self.agent_move_dist * math.sin(angle_rad)
            new_x = self.state.agent_x + delta_x
            new_y = self.state.agent_y + delta_y

            if self._is_within_bounds(new_x, new_y):
                self.state.agent_x, self.state.agent_y = int(new_x), int(new_y)
                print(f"Agent moves to ({self.state.agent_x}, {self.state.agent_y})")
            else:
                print("Agent move blocked by wall. Position unchanged.")
            player_turn_ended = True

        elif isinstance(action, AttackAction):
            target_threat = next((t for t in self.state.threats if t.id == action.target_id), None)
            
            if not target_threat:
                print(f"Attack failed: Threat with ID {action.target_id} not found.")
            else:
                distance = self._get_distance(self.state.agent_x, self.state.agent_y, target_threat.x, target_threat.y)
                
                if distance > self.agent_move_dist:
                    print(f"Attack failed: Threat {target_threat.id} is out of range.")
                elif self.state.agent_strength <= target_threat.strength:
                    print(f"Attack failed: Agent strength ({self.state.agent_strength}) is not greater than Threat {target_threat.id} strength ({target_threat.strength}).")
                else:
                    print(f"Agent attacks and defeats Threat {target_threat.id}!")
                    self.state.agent_x, self.state.agent_y = target_threat.x, target_threat.y
                    self.state.threats = [t for t in self.state.threats if t.id != action.target_id]
                    if not self.state.threats:
                        self.state.game_over = True
                        self.state.player_won = True
                        print("All threats defeated! YOU WIN!")

            player_turn_ended = True

        # After player's action, threats take their turn
        if player_turn_ended and not self.state.game_over:
            self._process_threat_turns()
1