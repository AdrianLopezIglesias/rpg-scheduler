# simulation/actions.py

from dataclasses import dataclass

class Action:
    """Base class for all actions."""
    pass

class HoldAction(Action):
    """The agent chooses to do nothing."""
    pass

@dataclass
class MoveAction(Action):
    """The agent chooses to move at a specific angle."""
    angle: float # Angle in degrees

@dataclass
class AttackAction(Action):
    """The agent chooses to attack a target."""
    target_id: int
