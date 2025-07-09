# main.py

import yaml
from simulation.simulator import Simulator
from simulation.actions import MoveAction, HoldAction

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
    print("Commands: 'move [angle]', 'hold', 'quit'")
    
    while True:
        print("-" * 20)
        print(f"Current Position: ({sim.state.agent_x}, {sim.state.agent_y})")
        
        user_input = input("Enter command: ").strip().lower()
        parts = user_input.split()
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
        else:
            print("Unknown command.")

if __name__ == "__main__":
    play_mode()

