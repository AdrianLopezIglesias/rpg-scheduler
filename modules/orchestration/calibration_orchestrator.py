import json
import copy
from ..utils import log
from .training_orchestrator import run_training_loop


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