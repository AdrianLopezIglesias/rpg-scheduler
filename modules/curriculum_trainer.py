import copy
import json
import os
import random
from .rl_trainer import run_rl_training
from .validator import run_validation
from .utils import log

def run_curriculum_training(config):
    curriculum_cfg = config['curriculum_config']
    log("=============== STARTING CURRICULUM TRAINING RUN ===============")
    
    maps_path = os.path.join(os.path.dirname(__file__), '..', 'game', 'maps.json')
    with open(maps_path, 'r') as f:
        all_maps = json.load(f)

    training_maps = [m for m in all_maps if m.get("used_for_training")]
    
    last_successful_model_path = None
    
    active_difficulties = []
    for current_map in training_maps:
        difficulty = current_map["id"]
        active_difficulties.append(difficulty)
        map_title = current_map.get('title', 'No Title Found')
        log(f"\n{'='*20} Starting Stage: {map_title} (Difficulties: {active_difficulties}) {'='*20}")
        
        target_win_rate = current_map['expected_winning_rate']
        model_to_load_for_this_stage = last_successful_model_path

        for i in range(curriculum_cfg['max_retries']):
            log(f"\n--- Attempt {i+1}/{curriculum_cfg['max_retries']} for Difficulty Stage {difficulty} ---")
            
            temp_config = copy.deepcopy(config)
            current_learning_rate = curriculum_cfg['learning_rate']
            if i > 0:
                current_learning_rate = current_learning_rate * 0.8
     
            log(f"This is a retry. Using smaller learning rate: {current_learning_rate}")

            model_save_path = f"models/{curriculum_cfg['model_name_prefix']}_diff_{difficulty}_attempt_{i+1}.pth"
            rl_config = {
                "difficulties": active_difficulties, 
                "learning_rate": current_learning_rate,
                "gamma": curriculum_cfg['gamma'],
          
               "num_episodes": curriculum_cfg['num_episodes'],
                "log_interval": curriculum_cfg['log_interval'],
                "load_model_path": model_to_load_for_this_stage,
                "model_save_path": model_save_path
            }
            temp_config['rl_config'] = rl_config
            run_rl_training(temp_config)
    
         
            model_to_load_for_this_stage = model_save_path

            validation_config = {
                "difficulty": difficulty,
                "model_path": model_save_path,
         
               "num_games": curriculum_cfg['validation_games']
            }
            temp_config['validation_config'] = validation_config
            val_results = run_validation(temp_config)
            
            current_win_rate = val_results.get("win_rate_percent", 0)
  
            avg_win = val_results.get("avg_win_speed", 'N/A')
            fastest_win = val_results.get("fastest_win_actions", 'N/A')

            win_rate_ok = current_win_rate >= target_win_rate
            speed_ok= True

            log(f"--- Validation Check for Attempt {i+1} ---")
            log(f"Target Win Rate: >={target_win_rate}%. Actual: {current_win_rate:.2f}%. -> {'MET' if win_rate_ok else 'NOT MET'}")
            
            if win_rate_ok and speed_ok:
                log(f"SUCCESS: Model passed all checks for difficulty {difficulty}.")
                last_successful_model_path = model_save_path
                break
        
            else:
                log("Conditions not met. Retraining...")
        else:
            log(f"\nFAILURE: Could not meet targets for difficulty {difficulty} within {curriculum_cfg['max_retries']} retries. Stopping curriculum.")
            return

    log("\n=============== CURRICULUM TRAINING FINISHED SUCCESSFULLY ===============")