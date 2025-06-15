import copy
from .rl_trainer import run_rl_training
from .validator import run_validation
from .utils import log

def run_curriculum_training(config):
    curriculum_cfg = config['curriculum_config']
    log("=============== STARTING CURRICULUM TRAINING RUN ===============")
    
    last_model_path = None

    for difficulty in curriculum_cfg['difficulties']:
        log(f"\n{'='*20} Starting Stage: Difficulty {difficulty} {'='*20}")
        target_win_rate = curriculum_cfg['targets'][difficulty]['target_win_rate']
        
        for i in range(curriculum_cfg['max_retries']):
            log(f"\n--- Attempt {i+1}/{curriculum_cfg['max_retries']} for Difficulty {difficulty} ---")
            
            # 1. TRAIN
            temp_config = copy.deepcopy(config)
            model_save_path = f"models/{curriculum_cfg['model_name_prefix']}_diff_{difficulty}_attempt_{i+1}.pth"
            rl_config = {
                "difficulty": difficulty,
                "learning_rate": curriculum_cfg['learning_rate'],
                "gamma": curriculum_cfg['gamma'],
                "num_episodes": curriculum_cfg['num_episodes'],
                "log_interval": curriculum_cfg['log_interval'],
                "load_model_path": last_model_path,
                "model_save_path": model_save_path
            }
            temp_config['rl_config'] = rl_config
            run_rl_training(temp_config)
            
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
            avg_win = val_results.get("avg_win_speed", float('inf'))
            fastest_win = val_results.get("fastest_win_actions", float('inf'))

            win_rate_ok = current_win_rate >= target_win_rate
            
            # Handle cases where there are no wins
            if fastest_win == 'N/A' or avg_win == 'N/A':
                speed_ok = False
            else:
                speed_ok = avg_win <= (fastest_win * 2)

            log(f"--- Validation Check for Attempt {i+1} ---")
            log(f"Target Win Rate: >={target_win_rate}%. Actual: {current_win_rate:.2f}%. -> {'MET' if win_rate_ok else 'NOT MET'}")
            log(f"Speed Target: Avg <= Fastest * 2. Actual: {avg_win:.2f} <= {fastest_win * 2:.2f}. -> {'MET' if speed_ok else 'NOT MET'}")

            if win_rate_ok and speed_ok:
                log(f"SUCCESS: Model passed all checks for difficulty {difficulty}.")
                last_model_path = model_save_path
                break
            else:
                log("Conditions not met. Retraining...")
        else:
            log(f"\nFAILURE: Could not meet targets for difficulty {difficulty} within {curriculum_cfg['max_retries']} retries. Stopping curriculum.")
            return

    log("\n=============== CURRICULUM TRAINING FINISHED SUCCESSFULLY ===============")