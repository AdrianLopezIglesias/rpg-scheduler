import os
import json
from pandemic_game import PandemicGame
from agents.agents import RandomAgent, NNAgent

def run_simulation(agent, num_games, output_path):
    """
    Runs a specified number of games using a given agent and saves the data.
    The data is saved as a list of games, where each game is a list of turns.
    """
    game = PandemicGame()
    all_games_data = [] # Changed from all_game_data to all_games_data for clarity
    
    print(f"Running {num_games} games with {agent.__class__.__name__}...")

    for i in range(num_games):
        game_history = []
        state = game.reset()
        
        while True:
            game_over_status = game.is_game_over()
            if game_over_status:
                # Add the final result to all steps in this game's history
                for step in game_history:
                    step["final_result"] = game_over_status
                    step["final_fitness"] = game.fitness_score
                break

            possible_actions = game.get_possible_actions()
            chosen_action = agent.choose_action(state, possible_actions)
            
            # Log the state before the action
            game_history.append({
                "state": state,
                "action_taken": chosen_action
            })
            
            # Take the step
            state = game.step(chosen_action)

        all_games_data.append(game_history) # Changed from .extend() to .append()
        if (i + 1) % 100 == 0:
            print(f"  ...completed {i + 1}/{num_games} games.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_games_data, f, indent=2)
    
    print(f"Simulation data saved to {output_path}")

# This part is now better handled by the main evolution loop,
# but can be kept for testing a single run.
if __name__ == "__main__":
    GENERATION = 0
    NUM_GAMES_TO_SIMULATE = 1000

    if GENERATION == 0:
        current_agent = RandomAgent()
    else:
        model_prefix = f"models/generation_{GENERATION - 1}/pandemic_model"
        current_agent = NNAgent(model_prefix)

    output_file = f"data/generation_{GENERATION}/simulation_data.json"
    
    run_simulation(agent=current_agent, num_games=NUM_GAMES_TO_SIMULATE, output_path=output_file)
