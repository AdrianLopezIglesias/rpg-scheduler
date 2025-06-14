import random
from .base_agent import Agent

class RandomAgent(Agent):
    """An agent that chooses a legal action randomly."""
    def choose_action(self, game, possible_actions):
        move_actions = [a for a in possible_actions if a['type'] == 'move']
        treat_actions = [a for a in possible_actions if a['type'] == 'treat']

        available_action_types = []
        if move_actions:
            available_action_types.append('move')
        if treat_actions:
            available_action_types.append('treat')

        if not available_action_types:
            return random.choice(possible_actions) # Fallback for 'pass' or empty list

        chosen_type = random.choice(available_action_types)

        if chosen_type == 'move':
            return random.choice(move_actions)
        else: # chosen_type == 'treat'
            return random.choice(treat_actions)