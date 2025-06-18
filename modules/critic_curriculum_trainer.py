# modules/critic_curriculum_trainer.py

import copy
import json
import os
import torch
import shutil
import glob
from .utils import log
from .validator import run_validation
from .mcts_dataset_generator import generate_training_data
from .critic_trainer import train_critic_from_data
from agents.agents import GNN_MCTS_Agent, RandomAgent, GNNAgent
from game.pandemic_game import PandemicGame
import random

import copy
import json
import os
import torch
import shutil
import glob
from .utils import log
from .validator import run_validation
from .mcts_dataset_generator import generate_training_data
from .critic_trainer import train_critic_from_data
from agents.agents import GNN_MCTS_Agent, RandomAgent, GNNAgent
from game.pandemic_game import PandemicGame
import random

def _load_all_training_data(data_dir):
    """Loads all .pt data files from a directory and combines them."""
    all_data = []
    all_files = glob.glob(os.path.join(data_dir, "*.pt"))
    log(f"Loading {len(all_files)} previous data files...")
    for f_path in all_files:
        # --- MODIFIED: Added weights_only=False to allow loading custom data objects ---
        all_data.extend(torch.load(f_path, weights_only=False))
    return all_data

def _pick_agent_for_data_gen(model_path, config):
    """Selects and logs the agent class for data generation."""
    curriculum_cfg = config['mcts_curriculum_config']
    random_prob = curriculum_cfg.get("random_agent_generation_prob", 0.0)
    agent_class_to_use = GNN_MCTS_Agent

    if not model_path:
        agent_class_to_use = RandomAgent
    elif random.random() < random_prob:
        agent_class_to_use = RandomAgent
    
    log_message = f"Data Generation Agent: '{agent_class_to_use.__name__}'"
    if agent_class_to_use == GNN_MCTS_Agent:
        log_message += f" (Model: {model_path})"
    log(log_message)

    return agent_class_to_use

def run_critic_curriculum(config):
    curriculum_cfg = config['mcts_curriculum_config']
    log("=============== STARTING CRITIC TRAINING CURRICULUM ===============")

    data_dir = "data/mcts_run_data/"
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)
    log(f"Created clean data directory for this run at: {data_dir}")

    last_successful_model_path = None
    active_difficulties = []

    for difficulty in curriculum_cfg['difficulties']:
        active_difficulties.append(difficulty)
        log(f"\n{'='*20} Starting Stage: (Difficulties: {active_difficulties}) {'='*20}")
        model_to_load_for_this_stage = last_successful_model_path

        for i in range(curriculum_cfg['max_retries']):
            num_random_games = curriculum_cfg.get("exploratory_random_games", 0)
            if num_random_games > 0:
                log(f"--- Generating {num_random_games} exploratory games for new difficulty '{difficulty}' ---")
                exploratory_output_path = os.path.join(data_dir, f"exploratory_diff_{difficulty}.pt")
                generate_training_data(
                    agent_class=RandomAgent,
                    model_path=None,
                    difficulties=[difficulty],
                    num_games=num_random_games,
                    config=config,
                    output_path=exploratory_output_path
                )

            target_win_rate = curriculum_cfg['targets'][difficulty]['target_win_rate']
            log(f"\n--- Attempt {i+1}/{curriculum_cfg['max_retries']} for Difficulty Stage {difficulty} ---")
            
            agent_for_data_gen = _pick_agent_for_data_gen(model_to_load_for_this_stage, config)
            
            num_games_for_generation = curriculum_cfg['games_per_generation'] * int(difficulty)
            output_path = os.path.join(data_dir, f"diff_{difficulty}_attempt_{i+1}.pt")

            generate_training_data(
                agent_class=agent_for_data_gen,
                model_path=model_to_load_for_this_stage,
                difficulties=[difficulty],
                num_games=num_games_for_generation,
                config=config,
                output_path=output_path
            )
            generate_training_data(
                agent_class=agent_for_data_gen,
                model_path=model_to_load_for_this_stage,
                difficulties=active_difficulties,
                num_games=num_games_for_generation,
                config=config,
                output_path=output_path
            )

            aggregated_training_data = _load_all_training_data(data_dir)

            temp_config = copy.deepcopy(config)
            temp_env = PandemicGame(difficulty=difficulty, config=config)
            input_dim = temp_env.get_node_feature_count()
            training_agent = GNNAgent(input_dim, temp_config)
            
            if model_to_load_for_this_stage:
                try:
                    training_agent.load_model(model_to_load_for_this_stage)
                except (FileNotFoundError, RuntimeError) as e:
                    log(f"Warning: Could not load model at {model_to_load_for_this_stage}. Training new model. Error: {e}")

            train_critic_from_data(training_agent, aggregated_training_data, temp_config,int(difficulty))
            
            model_save_path = f"models/{curriculum_cfg['model_name_prefix']}_diff_{difficulty}_attempt_{i+1}.pth"
            training_agent.save_model(model_save_path)
            log(f"Saved candidate model to {model_save_path}")
            model_to_load_for_this_stage = model_save_path
            
            validation_config = { "difficulty": difficulty, "model_path": model_save_path }
            if 'validation_config' in config and 'num_games' in config['validation_config']:
                validation_config['num_games'] = config['validation_config']['num_games']
            
            temp_config['validation_config'] = validation_config
            val_results = run_validation(temp_config)
            
            current_win_rate = val_results.get("win_rate_percent", 0)
            win_rate_ok = current_win_rate >= target_win_rate
            log(f"--- Validation Check ---")
            log(f"Target Win Rate: >={target_win_rate}%. Actual: {current_win_rate:.2f}%. -> {'MET' if win_rate_ok else 'NOT MET'}")
            if win_rate_ok:
                log(f"SUCCESS ✅✅✅✅✅✅ : Model passed all checks for difficulty {difficulty}.")
                last_successful_model_path = model_save_path
                break
            else:
                log("Conditions not met. Retrying with fine-tuned model...")
        else:
            log(f"\nFAILURE: Could not meet targets for difficulty {difficulty} within {curriculum_cfg['max_retries']} retries. Stopping curriculum.")
            return

    log("\n=============== CRITIC TRAINING CURRICULUM FINISHED SUCCESSFULLY ===============")


# You also need the _pick_agent_for_data_gen function in this file
def _pick_agent_for_data_gen(model_path, config):
    curriculum_cfg = config['mcts_curriculum_config']
    random_prob = curriculum_cfg.get("random_agent_generation_prob", 0.0)
    agent_class_to_use = GNN_MCTS_Agent
    if not model_path:
        agent_class_to_use = RandomAgent
    elif random.random() < random_prob:
        agent_class_to_use = RandomAgent
    log_message = f"Data Generation Agent: '{agent_class_to_use.__name__}'"
    if agent_class_to_use == GNN_MCTS_Agent:
        log_message += f" (Model: {model_path})"
    log(log_message)
    return agent_class_to_use