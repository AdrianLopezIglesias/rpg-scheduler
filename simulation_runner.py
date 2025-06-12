import os
import json
from pandemic_game import PandemicGame
from agents.agents import RandomAgent, NNAgent

def run_simulation(agent, num_games, output_path):
    """
    Runs games and saves them in the new format: a list of full game objects.
    """
    game = PandemicGame()
    all_games_played = []
    
    print(f"Running {num_games} games with {agent.__class__.__name__}...")

    for i in range(num_games):
        game_history = []
        state = game.reset()
        
        for _ in range(game.max_actions_per_game):
            game_over_status = game.is_game_over()
            if game_over_status:
                break

            possible_actions = game.get_possible_actions()
            chosen_action = agent.choose_action(state, possible_actions)
            
            # Log the state *before* the action is taken
            game_history.append({
                "state": state,
                "action_taken": chosen_action
            })
            
            state = game.step(chosen_action)

        # After the game is over, save the entire game as one object
        all_games_played.append({
            "game_history": game_history,
            "final_result": game.is_game_over(),
            "total_actions": game.actions_taken
        })
        
        if (i + 1) % 100 == 0:
            print(f"  ...completed {i + 1}/{num_games} games.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_games_played, f, indent=2)
    
    print(f"Simulation data saved to {output_path}")
