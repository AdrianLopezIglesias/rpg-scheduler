import json
import random
import os

# --- Core Game Logic Functions (Unchanged) ---
# These functions define the rules of our game world.

def calculate_time_to_train(player_level):
    """Calculates the time cost for the 'train' action."""
    return player_level * 2

def calculate_time_to_forge(player_element_level):
    """Calculates the time cost for the 'forgeElement' action."""
    return player_element_level

def is_ready_for_next_chapter(state):
    """Checks if the player's level is sufficient to advance the story."""
    player_level = state['playerLevel']
    element_level = state['playerElementLevel'][state['nextChapterElement']]
    next_chapter_level = state['nextChapterLevel']
    return (player_level + element_level * 2) > next_chapter_level

def determine_next_action(state):
    """Determines the optimal next action based on the game state logic."""
    if is_ready_for_next_chapter(state):
        return "advanceStory"
    else:
        time_to_train = calculate_time_to_train(state['playerLevel'])
        element_to_forge = state['nextChapterElement']
        element_level = state['playerElementLevel'][element_to_forge]
        time_to_forge = calculate_time_to_forge(element_level)

        if time_to_train <= time_to_forge:
            return "train"
        else:
            return f"forgeElement_{element_to_forge}"

# --- NEW: Scenario-Based Generation Functions ---
# Instead of pure random generation, we create specific scenarios to ensure each action type is represented.

def generate_advance_story_scenario(elements):
    """Generates a state where advancing the story is the correct action."""
    player_level = random.randint(10, 20)
    next_chapter_element = random.choice(elements)
    player_element_level = {el: random.randint(10, 20) for el in elements}
    
    # Ensure the player is over-levelled for the next chapter
    required_level = player_level + player_element_level[next_chapter_element] * 2
    next_chapter_level = random.randint(required_level - 15, required_level - 5)
    
    return {
        "playerLevel": player_level,
        "playerElementLevel": player_element_level,
        "nextChapterLevel": next_chapter_level,
        "nextChapterElement": next_chapter_element
    }

def generate_train_scenario(elements):
    """Generates a state where training is the most time-efficient action."""
    # To make training cheaper, playerLevel should be low
    player_level = random.randint(1, 5)
    next_chapter_element = random.choice(elements)
    
    # To make forging expensive, the required element level should be high
    player_element_level = {el: random.randint(1, 10) for el in elements}
    player_element_level[next_chapter_element] = random.randint(15, 20)
    
    # Ensure the player is not ready for the next chapter yet
    required_level = player_level + player_element_level[next_chapter_element] * 2
    next_chapter_level = random.randint(required_level + 5, required_level + 15)

    return {
        "playerLevel": player_level,
        "playerElementLevel": player_element_level,
        "nextChapterLevel": next_chapter_level,
        "nextChapterElement": next_chapter_element
    }

def generate_forge_scenario(elements):
    """Generates a state where forging is the most time-efficient action."""
    # To make training expensive, playerLevel should be high
    player_level = random.randint(15, 20)
    next_chapter_element = random.choice(elements)
    
    # To make forging cheap, the required element level should be low
    player_element_level = {el: random.randint(1, 10) for el in elements}
    player_element_level[next_chapter_element] = random.randint(1, 5)

    # Ensure the player is not ready for the next chapter yet
    required_level = player_level + player_element_level[next_chapter_element] * 2
    next_chapter_level = random.randint(required_level + 5, required_level + 15)

    return {
        "playerLevel": player_level,
        "playerElementLevel": player_element_level,
        "nextChapterLevel": next_chapter_level,
        "nextChapterElement": next_chapter_element
    }


def generate_synthetic_data(num_samples=1000):
    """
    Generates a balanced dataset by creating a specific number of samples for each action type.
    """
    print(f"Generating {num_samples} balanced data samples...")
    data = []
    elements = ["fire", "water", "earth", "wind"]
    
    # Define the proportion of the dataset for each scenario
    proportions = {
        "advance": 0.4,
        "train": 0.3,
        "forge": 0.3
    }
    
    for _ in range(int(num_samples * proportions["advance"])):
        data.append(generate_advance_story_scenario(elements))
        
    for _ in range(int(num_samples * proportions["train"])):
        data.append(generate_train_scenario(elements))

    for _ in range(int(num_samples * proportions["forge"])):
        data.append(generate_forge_scenario(elements))

    # The list 'data' now contains states tailored to specific outcomes.
    # We now determine the correct action for each generated state to create the final dataset.
    final_data = []
    for state in data:
        action = determine_next_action(state)
        final_data.append({"gameState": state, "nextAction": action})

    # Shuffle the data to ensure randomness
    random.shuffle(final_data)

    output_path = os.path.join(os.path.dirname(__file__), '..', 'synthetic_data.json')
        
    with open(output_path, "w") as f:
        json.dump(final_data, f, indent=2)
        
    print(f"Balanced synthetic data generated and saved to {output_path}")

if __name__ == "__main__":
    generate_synthetic_data(5000) # Generating 5000 samples as an example
