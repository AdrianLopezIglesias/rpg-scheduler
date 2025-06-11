import json
import random

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
    """Determines the optimal next action based on the game state."""
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

def generate_game_state(elements):
    """Generates a random game state."""
    player_level = random.randint(1, 20)
    next_chapter_level = player_level + random.randint(3, 10)
    
    player_element_level = {el: random.randint(1, 10) for el in elements}
    next_chapter_element = random.choice(elements)

    return {
        "playerLevel": player_level,
        "playerElementLevel": player_element_level,
        "nextChapterLevel": next_chapter_level,
        "nextChapterElement": next_chapter_element
    }

def generate_synthetic_data(num_samples=1000):
    """Generates a dataset of game states and their corresponding optimal actions."""
    print(f"Generating {num_samples} data samples...")
    data = []
    elements = ["fire", "water", "earth", "wind"]

    for _ in range(num_samples):
        state = generate_game_state(elements)
        action = determine_next_action(state)
        data.append({"gameState": state, "nextAction": action})
        
    with open("synthetic_data.json", "w") as f:
        json.dump(data, f, indent=2)
        
    print("Synthetic data generated and saved to synthetic_data.json")

if __name__ == "__main__":
    generate_synthetic_data()
