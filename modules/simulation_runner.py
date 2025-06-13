import os
import json
from game.pandemic_game import PandemicGame
from agents.agents import RandomAgent, NNAgent

def run_simulation(agent, num_games, output_path, difficulty, config):
    """
    Runs a specified number of games using a given agent and saves the data.
    Now passes the config object to the game instance.
    """
    game = PandemicGame(difficulty=difficulty, config=config)
    all_games_played = []
    
    print(f"Running {num_games} games with {agent.__class__.__name__} on '{difficulty}' map...")

    for i in range(num_games):
        game_history = []
        state = game.reset()
        
        while True:
            game_over_status = game.is_game_over()
            if game_over_status:
                break

            possible_actions = game.get_possible_actions()
            chosen_action = agent.choose_action(game, possible_actions)
            
            game_history.append({
                "state": state,
                "action_taken": chosen_action
            })
            
            state = game.step(chosen_action)

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
