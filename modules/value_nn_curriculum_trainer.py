import os
import shutil
import glob
import torch
import copy
from .utils import log
from .value_nn_validator import run_value_nn_validation
from .value_nn_dataset_generator import generate_value_nn_data
from .value_nn_trainer import train_value_nn
from agents.agents import RandomAgent
from agents.value_nn_agent import ValueNNAgent, ValueNet
from game.pandemic_game_extended import PandemicGameExtended

def _load_all_value_data(data_dir):
    """Loads all .pt files from a directory into a single list."""
    all_data = []
    all_files = glob.glob(os.path.join(data_dir, "*.pt"))
    if not all_files:
        return all_data
    
    log(f"🗂️  Loading {len(all_files)} data files from previous attempts...")
    for f_path in all_files:
        try:
            all_data.extend(torch.load(f_path, weights_only=False))
        except (IOError, EOFError) as e:
            log(f"⚠️  Warning: Could not load data from {f_path}. File might be corrupt. Skipping. Error: {e}")
    return all_data

def _train_candidate_model(input_dim, data_dir, model_to_load, config, difficulty, attempt):
    """Loads all data, trains a model, and saves it."""
    log(f"--- 🧠 Attempt {attempt}: Training Phase ---")
    
    aggregated_training_data = _load_all_value_data(data_dir)
    log(f"    📚 Total training samples available: {len(aggregated_training_data)}")
    
    if not aggregated_training_data:
        log("    ⚠️ No training data found. Skipping model training.")
        return model_to_load
        
    training_model = ValueNet(input_dim)
    if model_to_load:
        try:
            training_model.load_state_dict(torch.load(model_to_load))
            log(f"    ☑️  Loaded weights from '{os.path.basename(model_to_load)}' into new model.")
        except Exception as e:
            log(f"    ⚠️  Could not load model weights, starting fresh. Error: {e}")

    train_value_nn(training_model, aggregated_training_data, config)
    
    candidate_model_path = f"models/value_nn_diff_{difficulty}_attempt_{attempt}.pth"
    os.makedirs(os.path.dirname(candidate_model_path), exist_ok=True)
    torch.save(training_model.state_dict(), candidate_model_path)
    log(f"    💾 Saved candidate model to '{candidate_model_path}'")
    return candidate_model_path

def run_value_nn_curriculum(config):
    curriculum_cfg = config['value_nn_curriculum_config']
    log("🚀=============== STARTING VALUE NET (NN) CURRICULUM ===============🚀")

    # Clean and prepare directories for the new run
    model_dir = "models/"
    data_dir = "data/value_nn_run_data/"
    if os.path.exists(data_dir): shutil.rmtree(data_dir)
    if os.path.exists(model_dir): shutil.rmtree(model_dir)
    os.makedirs(data_dir)
    os.makedirs(model_dir)
    log(f"✅ Created clean data and model directories for this run.")

    last_successful_model_path = None
    
    temp_env = PandemicGameExtended(difficulty=curriculum_cfg['difficulties'][0], config=config)
    input_dim = temp_env.get_feature_vector_size()
    del temp_env
    
    active_difficulties = []

    for difficulty in curriculum_cfg['difficulties']:
        active_difficulties.append(difficulty)
        log(f"\n{'='*20} 🏁 Starting Stage: Difficulty {difficulty} (Active Maps: {active_difficulties}) {'='*20}")
        
        # Clear previous data at the start of a new difficulty stage
        if os.path.exists(data_dir): shutil.rmtree(data_dir)
        os.makedirs(data_dir)
        log(f"   Cleared data directory for new difficulty stage.")
        
        model_to_load_for_this_stage = last_successful_model_path

        for i in range(curriculum_cfg['max_retries']):
            attempt_num = i + 1
            log(f"\n--- 🔄 Cycle {attempt_num}/{curriculum_cfg['max_retries']} for Difficulty Stage {difficulty} ---")
            
            log(f"🤖 Cycle {attempt_num}.1: Generating new data...")
            # Generate random data to ensure exploration
            exploratory_output_path = os.path.join(data_dir, f"data_exploratory_attempt_{attempt_num}.pt")
            generate_value_nn_data(
                agent_class=RandomAgent, model_path=None, input_dim=input_dim,
                difficulties=[difficulty], num_games=curriculum_cfg['exploratory_games'],
                config=config, output_path=exploratory_output_path
            )

            # Train a model on all data gathered so far in this stage
            candidate_model_path = _train_candidate_model(
                input_dim=input_dim, data_dir=data_dir,
                model_to_load=model_to_load_for_this_stage,
                config=config, difficulty=difficulty, attempt=attempt_num
            )

            # Generate more data using the newly trained agent
            agent_data_output_path = os.path.join(data_dir, f"data_agent_attempt_{attempt_num}.pt")
            generate_value_nn_data(
                agent_class=ValueNNAgent, model_path=candidate_model_path, input_dim=input_dim,
                difficulties=active_difficulties, num_games=curriculum_cfg['games_per_generation'],
                config=config, output_path=agent_data_output_path
            )

            log(f"\n🤖 Cycle {attempt_num}.2: Validating model performance...")
            val_results = run_value_nn_validation(config, candidate_model_path, difficulty)
            current_win_rate = val_results.get("win_rate_percent", 0)
            
            target_win_rate = curriculum_cfg['targets'][difficulty]['target_win_rate']
            win_rate_ok = current_win_rate >= target_win_rate

            log(f"\n--- 📊 Validation Check for Cycle {attempt_num} ---")
            log(f"Target Win Rate: >={target_win_rate}%. Actual: {current_win_rate:.2f}%. -> {'✅ MET' if win_rate_ok else '❌ NOT MET'}")
            
            if win_rate_ok:
                log(f"🎉 SUCCESS! Model passed all checks for difficulty {difficulty}. Moving to next stage! 🎉")
                last_successful_model_path = candidate_model_path
                break
            else:
                log("   Conditions not met. Retrying with fine-tuned model...")
                model_to_load_for_this_stage = candidate_model_path 
        else:
            log(f"\n🛑 FAILURE: Could not meet targets for difficulty {difficulty} after {curriculum_cfg['max_retries']} retries. Stopping curriculum. 🛑")
            return

    log("\n\n🏆=============== VALUE NET (NN) CURRICULUM FINISHED SUCCESSFULLY ===============🏆")
