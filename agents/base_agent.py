class Agent:
    """A base class for all agents, defining the required interface."""
    def choose_action(self, game, possible_actions):
        """
        All agent subclasses must implement this method.
        
        Args:
            game: The current game instance.
            possible_actions: A list or mask of legal actions.

        Returns:
            The chosen action.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")