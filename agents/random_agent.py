import random
from .base_agent import Agent

class RandomAgent(Agent):
    """An agent that chooses any legal action randomly, with a bias against passing."""
    def choose_action(self, game, possible_actions):
        """
        Chooses a random legal action. It will only 'pass' if it's the only option.
        """
        if not possible_actions:
            return None

        
        # If only 'pass' is available, return it
        return random.choice(possible_actions)
