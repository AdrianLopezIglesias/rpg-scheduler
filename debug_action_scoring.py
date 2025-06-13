import numpy as np
from game.pandemic_game import PandemicGame
from agents.agents import NNAgent, RandomAgent, create_feature_vector

def debug_decision(game, agent):
    """Runs one turn and prints the agent's decision-making process."""
    
    state = game.get_state_snapshot()
    player_loc = state['player_location']
    print(f"\n--- Turn {state['actions_taken']} | Player at: {player_loc} ---")
    
    # 1. Show the input feature vector
    feature_vector = create_feature_vector(state, game)
    print(f"  Input Vector (to model): {feature_vector}")

    if isinstance(agent, NNAgent) and agent.model:
        # 2. Show the raw output from the model
        scaled_features = agent.scaler.transform(feature_vector)
        action_scores = agent.model.predict(scaled_features)[0]
        print(f"  Model Output (raw scores): {np.round(action_scores, 2)}")

        # 3. Show how the agent maps scores to possible actions
        print("  Decision Process:")
        possible_actions = game.get_possible_actions()
        sorted_neighbors = sorted(game.map[player_loc]['neighbors'])
        
        # Check 'treat' action
        score_treat = action_scores[0]
        is_legal = {"type": "treat", "target": player_loc} in possible_actions
        print(f"    - Action 'treat': Score={score_treat:.2f} (Legal: {is_legal})")

        # Check 'move' actions
        for i in range(4):
            score_move = action_scores[i+1]
            if i < len(sorted_neighbors):
                neighbor = sorted_neighbors[i]
                is_legal = {"type": "move", "target": neighbor} in possible_actions
                print(f"    - Action 'move to {neighbor}': Score={score_move:.2f} (Legal: {is_legal})")
            else:
                 print(f"    - Action 'move to neighbor #{i+1}': Score={score_move:.2f} (Legal: False - no such neighbor)")

    # 4. Show the final decision
    chosen_action = agent.choose_action(game, game.get_possible_actions())
    print(f"  ==> FINAL DECISION: {chosen_action}")
    return chosen_action


if __name__ == "__main__":
    DIFFICULTY = "easy"
    MODEL_GENERATION = 1 #<-- Change this to test different trained models

    game = PandemicGame(difficulty=DIFFICULTY)
    
    # Load the agent to debug
    model_path_prefix = f"models/{DIFFICULTY}/generation_{MODEL_GENERATION}/pandemic_model"
    agent_to_debug = NNAgent(model_path_prefix, epsilon=0) # Epsilon=0 to force exploitation

    if not agent_to_debug.model:
        print("\nCould not load model. Running with RandomAgent instead.")
        agent_to_debug = RandomAgent()

    state = game.reset()
    
    print("\n===========================================")
    print(f"  DEBUGGING AGENT DECISION - GEN {MODEL_GENERATION}  ")
    print("===========================================\n")

    for _ in range(15):
        if game.is_game_over():
            break
        
        action = debug_decision(game, agent_to_debug)
        game.step(action)
