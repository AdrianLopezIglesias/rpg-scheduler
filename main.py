# main.py

import yaml
from simulation.simulator import Simulator
from simulation.actions import MoveAction, HoldAction, AttackAction

def load_config(path="config/sim_config.yaml"):
    """Loads the simulation configuration from a YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def play_mode():
    """
    Runs an interactive play mode in the console to test the simulator.
    """
    config = load_config()
    sim = Simulator(config)
    
    print("--- Simulation Play Mode ---")
    print("Commands: 'move [angle]', 'attack [id]', 'hold', 'quit'")
    
    while not sim.state.game_over:
        print("-" * 20)
        print(f"Player: Pos=({sim.state.agent_x}, {sim.state.agent_y}), Str={sim.state.agent_strength}")
        
        # Display threat locations and strengths
        if sim.state.threats:
            print("Threats:")
            for t in sim.state.threats:
                print(f"  - Threat {t.id}: Pos=({t.x}, {t.y}), Str={t.strength}")
        else:
            print("No threats remain.")

        user_input = input("Enter command: ").strip().lower()
        parts = user_input.split()
        if not parts:
            continue
        command = parts[0]

        if command == "quit":
            print("Exiting play mode.")
            break
        elif command == "hold":
            sim.step(HoldAction())
        elif command == "move":
            if len(parts) > 1:
                try:
                    angle = float(parts[1])
                    sim.step(MoveAction(angle=angle))
                except ValueError:
                    print("Invalid angle. Please enter a number.")
            else:
                print("Move command requires an angle. E.g., 'move 45'")
        elif command == "attack":
            if len(parts) > 1:
                try:
                    target_id = int(parts[1])
                    sim.step(AttackAction(target_id=target_id))
                except ValueError:
                    print("Invalid ID. Please enter a number.")
            else:
                print("Attack command requires a target ID. E.g., 'attack 0'")
        else:
            print("Unknown command.")

    if sim.state.game_over:
        if sim.state.player_won:
            print("\n--- Congratulations, you have won! ---")
        else:
            print("\n--- GAME OVER ---")

if __name__ == "__main__":
    play_mode()
