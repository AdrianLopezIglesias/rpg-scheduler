import json
import random
import os

# --- NEW: World and Game Logic with Logistics ---
ELEMENTS = ["fire", "water", "earth", "wind"]
LOCATIONS = {
    "City": {"purpose": "train", "travel_time": {"Volcano": 2, "Mines": 3, "Forest": 2, "Mountains": 4}},
    "Volcano": {"purpose": "forge_fire", "travel_time": {"City": 2, "Mines": 5, "Forest": 4, "Mountains": 6}},
    "Mines": {"purpose": "forge_earth", "travel_time": {"City": 3, "Volcano": 5, "Forest": 1, "Mountains": 2}},
    "Forest": {"purpose": "forge_wind", "travel_time": {"City": 2, "Volcano": 4, "Mines": 1, "Mountains": 3}},
    "Mountains": {"purpose": "forge_water", "travel_time": {"City": 4, "Volcano": 6, "Mines": 2, "Forest": 3}},
}
FORGE_LOCATIONS = {
    "fire": "Volcano",
    "water": "Mountains",
    "earth": "Mines",
    "wind": "Forest"
}
TRAIN_LOCATION = "City"

def calculate_time_to_train(player_level):
    return player_level * 2

def calculate_time_to_forge(player_element_level):
    return player_element_level

def is_ready_for_next_chapter(state):
    """
    New, more complex scoring. The primary element is worth 1.5x, others 0.25x.
    """
    player_level = state['playerLevel']
    primary_element = state['nextChapterElement']
    primary_level = state['playerElementLevel'][primary_element]
    
    other_elements_level = 0
    for el, lvl in state['playerElementLevel'].items():
        if el != primary_element:
            other_elements_level += lvl
            
    total_score = player_level + (primary_level * 1.5) + (other_elements_level * 0.25)
    return total_score > state['nextChapterLevel']

def determine_next_action(state):
    """
    Determines the optimal action, now including travel time as a cost.
    """
    player_location = state['playerLocation']

    # 1. Check if ready to advance and at the right location
    chapter_location = state['nextChapterLocation']
    if is_ready_for_next_chapter(state):
        if player_location == chapter_location:
            return "advanceStory"
        else:
            return f"travel_to_{chapter_location}"

    # 2. If not ready, calculate the cost of training vs. forging
    # Cost of Training = Travel time to City + Training Time
    travel_to_train_time = LOCATIONS[player_location]["travel_time"].get(TRAIN_LOCATION, 0) # 0 if already there
    train_cost = travel_to_train_time + calculate_time_to_train(state['playerLevel'])

    # Cost of Forging = Travel time to Forge Location + Forging Time
    element_to_forge = state['nextChapterElement']
    forge_location = FORGE_LOCATIONS[element_to_forge]
    travel_to_forge_time = LOCATIONS[player_location]["travel_time"].get(forge_location, 0)
    forge_cost = travel_to_forge_time + calculate_time_to_forge(state['playerElementLevel'][element_to_forge])

    # 3. Compare costs to decide primary goal (train or forge)
    if train_cost <= forge_cost:
        # Goal is to train. Are we at the training location?
        if player_location == TRAIN_LOCATION:
            return "train"
        else:
            return f"travel_to_{TRAIN_LOCATION}"
    else:
        # Goal is to forge. Are we at the right forge?
        if player_location == forge_location:
            return f"forgeElement_{element_to_forge}"
        else:
            return f"travel_to_{forge_location}"


# --- Scenario-Based Generation Functions (Updated for new logic) ---

def generate_advance_story_scenario():
    player_level = random.randint(15, 20)
    primary_element = random.choice(ELEMENTS)
    player_element_level = {el: random.randint(15, 20) for el in ELEMENTS}
    
    other_elements_level = sum(lvl for el, lvl in player_element_level.items() if el != primary_element)
    total_score = player_level + (player_element_level[primary_element] * 1.5) + (other_elements_level * 0.25)
    
    # Player is over-levelled, so next chapter should be lower
    next_chapter_level = random.uniform(total_score - 10, total_score - 2)
    chapter_location = random.choice(list(LOCATIONS.keys()))
    
    return {
        "playerLevel": player_level,
        "playerElementLevel": player_element_level,
        "nextChapterLevel": next_chapter_level,
        "nextChapterElement": primary_element,
        "playerLocation": chapter_location, # Player starts at the right spot
        "nextChapterLocation": chapter_location
    }

def generate_complex_scenario():
    """Generates a scenario where a decision between train/forge/travel is required."""
    player_level = random.randint(5, 15)
    primary_element = random.choice(ELEMENTS)
    player_element_level = {el: random.randint(1, 10) for el in ELEMENTS}

    other_elements_level = sum(lvl for el, lvl in player_element_level.items() if el != primary_element)
    total_score = player_level + (player_element_level[primary_element] * 1.5) + (other_elements_level * 0.25)
    
    # Player is under-levelled
    next_chapter_level = random.uniform(total_score + 5, total_score + 15)
    
    return {
        "playerLevel": player_level,
        "playerElementLevel": player_element_level,
        "nextChapterLevel": next_chapter_level,
        "nextChapterElement": primary_element,
        "playerLocation": random.choice(list(LOCATIONS.keys())),
        "nextChapterLocation": random.choice(list(LOCATIONS.keys()))
    }


def generate_synthetic_data(num_samples=1000):
    print(f"Generating {num_samples} complex data samples...")
    data = []
    
    proportions = {"advance": 0.3, "complex": 0.7}
    
    for _ in range(int(num_samples * proportions["advance"])):
        data.append(generate_advance_story_scenario())
        
    for _ in range(int(num_samples * proportions["complex"])):
        data.append(generate_complex_scenario())

    final_data = []
    for state in data:
        action = determine_next_action(state)
        final_data.append({"gameState": state, "nextAction": action})

    random.shuffle(final_data)
    output_path = os.path.join(os.path.dirname(__file__), '..', 'synthetic_data.json')
        
    with open(output_path, "w") as f:
        json.dump(final_data, f, indent=2)
        
    print(f"Complex synthetic data generated and saved to {output_path}")

if __name__ == "__main__":
    generate_synthetic_data(5000)
