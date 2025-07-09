# simulation/game_state.py

from dataclasses import dataclass, field
from typing import List

@dataclass
class Threat:
    """Represents a single threat in the game."""
    id: int
    x: int
    y: int
    strength: int

@dataclass
class GameState:
    """
    Represents the state of the game, including agent and threat properties.
    """
    agent_x: int
    agent_y: int
    agent_strength: int
    threats: List[Threat] = field(default_factory=list)
    game_over: bool = False
    player_won: bool = False
