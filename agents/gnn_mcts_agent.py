# agents/gnn_mcts_agent.py

import copy
import torch
import json
from .base_agent import Agent
from .policy_network import PolicyNetwork
from game.pandemic_game import PandemicGame

class GNN_MCTS_Agent(Agent):
    """
    An agent that uses a 1-step lookahead (a simple Monte Carlo Tree Search)
    to plan its moves. It uses a perfect simulator to find the next state
    and a pre-trained GNN Critic to evaluate the quality of that state.
    """
    def __init__(self, model_path, difficulty, config):
        # This agent needs a pre-trained model to function.
        # It will use the 'value_head' (Critic) from this network.
        temp_env = PandemicGame(difficulty=difficulty, config=config)
        input_dim = temp_env.get_node_feature_count()
        
        self.policy_network = PolicyNetwork(input_dim)
        if model_path:
            # --- MODIFIED LOADING LOGIC ---
            # Load the entire checkpoint dictionary.
            checkpoint = torch.load(model_path)
            # Extract and load only the model's state dictionary.
            self.policy_network.load_state_dict(checkpoint['model_state_dict'])
            # --- End of modification ---
            
        self.policy_network.eval() # Set the network to evaluation mode.

    def _simulate_next_state(self, game, action):
        """
        The "Perfect Transition Model" (NN1).
        Creates a hypothetical future state by applying an action to a copy
        of the current game state.
        """
        sim_game = copy.deepcopy(game)
        
        # The environment's step function requires an index, so we must convert our action dict back
        action_json = json.dumps(action, sort_keys=True)
        action_idx = -1
        for idx, action_dict in sim_game.idx_to_action.items():
            if json.dumps(action_dict, sort_keys=True) == action_json:
                action_idx = idx
                break

        if action_idx != -1:
            sim_game.step(action_idx)
            
        return sim_game

    def choose_action(self, game, possible_actions):
        """
        The planning loop. For each possible action, it simulates the resulting
        state, evaluates that state with the GNN Critic, and chooses the action
        that leads to the state with the highest predicted value.
        """
        best_action = None
        best_predicted_score = -float('inf')

        with torch.no_grad():
            for action in possible_actions:
                # 1. SIMULATE: Get the next game state using the perfect simulator.
                next_game_state = self._simulate_next_state(game, action)
                
                # 2. EVALUATE: Get the graph representation and score it with the Critic.
                state_graph = next_game_state.get_state_as_graph()
                state_graph.batch = torch.zeros(state_graph.num_nodes, dtype=torch.long)
                
                # We only need the third value returned, the state_value from the Critic.
                _, _, predicted_score = self.policy_network(state_graph)

                # 3. CHOOSE BEST: Keep track of the action that leads to the best-rated state.
                if predicted_score > best_predicted_score:
                    best_predicted_score = predicted_score
                    best_action = action
        
        return best_action if best_action else possible_actions[0]