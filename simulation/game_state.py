# simulation/game_state.py

from dataclasses import dataclass

@dataclass
class GameState:
    """
    Represents the state of the game, primarily the agent's position.
    """
    agent_x: int
    agent_y: int
