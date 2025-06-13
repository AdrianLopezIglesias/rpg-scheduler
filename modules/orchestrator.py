import json
import copy
import time
from .simulation_runner import run_simulation
from .trainer import train_next_generation
from .analysis import analyze_generation_data
from .utils import log
from agents.agents import RandomAgent, NNAgent

def run_training_loop(config):
    """
    Runs the main generational training loop for a single trial.
    """
    cfg = config['train_config']
    agent_cfg = config['agent_config']
    curriculum = cfg['curriculum']
    
    epsilon_decay = (agent_cfg['epsilon_start'] - agent_cfg['epsilon_end']) / len(curriculum) if curriculum else 0

    for gen, difficulty in enumerate(curriculum):
        log(f"=============== STARTING GENERATION {gen} on '{difficulty}' map ===============")
        sim_output_path = f"data/{difficulty}/generation_{gen}/simulation_data.json"
        start_time = time.time()

        if gen == 0:
            agent = RandomAgent()
        else:
            prev_difficulty = curriculum[gen - 1]
            model_path_prefix = f"models/{prev_difficulty}/generation_{gen}/pandemic_model"
            current_epsilon = agent_cfg['epsilon_start'] - (gen * epsilon_decay)
            agent = NNAgent(model_path_prefix, epsilon=current_epsilon)
            if not agent.model:
                log(f"Failed to load model for Gen {gen}. Halting.")
                break
        
        run_simulation(agent, cfg['games_per_generation'], sim_output_path, difficulty=difficulty, config=config)
        log(f"Simulation finished in {time.time() - start_time:.2f}s.")
        
        analysis_results = {}
        try:
            with open(sim_output_path, 'r') as f:
                analysis_results = analyze_generation_data(json.load(f))
        except FileNotFoundError:
            log(f"Could not find simulation data at {sim_output_path} to analyze.")
        
        start_time = time.time()
        train_next_generation(gen, difficulty, config, analysis_results)
        log(f"Training finished in {time.time() - start_time:.2f}s.")
        
    # Return final results for calibration
    return analysis_results if 'analysis_results' in locals() else {}

def run_calibration_loop(config):
    """
    Orchestrates the calibration process, testing multiple architectures.
    """
    log("=============== STARTING CALIBRATION RUN ===============")
    cal_cfg = config['calibrate_config']
    architectures = cal_cfg['architectures_to_test']
    trials = cal_cfg['trials_per_architecture']
    
    calibration_report = {}
    report_path = "reports/calibration_report.json"

    for arch in architectures:
        log(f"--- Testing Architecture: {arch} ---")
        arch_win_rates = []
        
        for i in range(trials):
            log(f"  Starting Trial {i + 1}/{trials} for architecture {arch}...")
            trial_config = copy.deepcopy(config)
            trial_config['model_config']['hidden_layer_sizes'] = arch

            final_generation_results = run_training_loop(trial_config)
            arch_win_rates.append(final_generation_results.get("win_rate_percent", 0))
            log(f"  Trial {i + 1} finished with a final win rate of: {arch_win_rates[-1]:.2f}%")

        calibration_report[str(arch)] = arch_win_rates
        
        with open(report_path, 'w') as f:
            json.dump(calibration_report, f, indent=4)
        log(f"Updated calibration report at {report_path}")

    log("=============== CALIBRATION RUN FINISHED ===============")
