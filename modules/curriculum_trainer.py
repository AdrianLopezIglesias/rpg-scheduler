import copy
import json
import os
from .rl_trainer import run_rl_training
from .validator import run_validation
from .utils import log

def run_curriculum_training(config):
    curriculum_cfg = config['curriculum_config']
    log("=============== STARTING CURRICULUM TRAINING RUN ===============")
    
    # Load map data to access titles
    maps_path = os.path.join(os.path.dirname(__file__), '..', 'game', 'maps.json')
    with open(maps_path, 'r') as f:
        maps_data = json.load(f)

    last_successful_model_path = None
    for difficulty in curriculum_cfg['difficulties']:
        map_title = maps_data.get(difficulty, {}).get('title', 'No Title Found')
        log(f"\n{'='*20} Starting Stage: Difficulty {difficulty} ({map_title}) {'='*20}")
        target_win_rate = curriculum_cfg['targets'][difficulty]['target_win_rate']
        
        # This will track the most recent model for the current difficulty, including failed attempts.
        model_to_load_for_this_stage = last_successful_model_path

        for i in range(curriculum_cfg['max_retries']):
            log(f"\n--- Attempt {i+1}/{curriculum_cfg['max_retries']} for Difficulty {difficulty} ---")
            
            # 1. TRAIN
            temp_config = copy.deepcopy(config)
            current_learning_rate = curriculum_cfg['learning_rate']
            if i > 0: # This is a retry
                current_learning_rate = current_learning_rate * 0.5
                log(f"This is a retry. Using smaller learning rate: {current_learning_rate}")

            model_save_path = f"models/{curriculum_cfg['model_name_prefix']}_diff_{difficulty}_attempt_{i+1}.pth"
            rl_config = {
                "difficulty": difficulty,
                "learning_rate": current_learning_rate, # Use the potentially adjusted learning rate
                "gamma": curriculum_cfg['gamma'],
                "num_episodes": curriculum_cfg['num_episodes'],
                "log_interval": curriculum_cfg['log_interval'],
                "load_model_path": model_to_load_for_this_stage, 
                "model_save_path": model_save_path
            }
            temp_config['rl_config'] = rl_config
            run_rl_training(temp_config)
            
            # The next attempt for THIS difficulty should load the model we just saved.
            model_to_load_for_this_stage = model_save_path

            # 2. VALIDATE
            validation_config = {
                "difficulty": difficulty,
                "model_path": model_save_path,
                "num_games": curriculum_cfg['validation_games']
            }
            temp_config['validation_config'] = validation_config
            val_results = run_validation(temp_config)

            # 3. CHECK CONDITIONS
            current_win_rate = val_results.get("win_rate_percent", 0)
            avg_win = val_results.get("avg_win_speed", 'N/A')
            fastest_win = val_results.get("fastest_win_actions", 'N/A')

            win_rate_ok = current_win_rate >= target_win_rate
            
            if fastest_win == 'N/A' or avg_win == 'N/A':
                speed_ok = False
            else:
                speed_ok = (avg_win) <= ((fastest_win  + 10) * 3)

            log(f"--- Validation Check for Attempt {i+1} ---")
            log(f"Target Win Rate: >={target_win_rate}%. Actual: {current_win_rate:.2f}%. -> {'MET' if win_rate_ok else 'NOT MET'}")
            
            if isinstance(avg_win, str) or isinstance(fastest_win, str):
                log("Speed Target: N/A (no wins recorded)")
            else:
                log(f"Speed Target: Avg <= ((Fastest + 10) * 3). Actual: {avg_win:.2f} <= {((fastest_win  + 10) * 3):.2f}. -> {'MET' if speed_ok else 'NOT MET'}")

            if win_rate_ok and speed_ok:
                log(f"SUCCESS: Model passed all checks for difficulty {difficulty}.")
                # This becomes the model to load for the NEXT difficulty.
                last_successful_model_path = model_save_path
                break
            else:
                log("Conditions not met. Retraining...")
        else:
            log(f"\nFAILURE: Could not meet targets for difficulty {difficulty} within {curriculum_cfg['max_retries']} retries. Stopping curriculum.")
            return

    log("\n=============== CURRICULUM TRAINING FINISHED SUCCESSFULLY ===============")