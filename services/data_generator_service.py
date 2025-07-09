import os
from .value_nn_dataset_generator import generate_value_nn_data
from agents.agents import RandomAgent
from agents.value_nn_agent import ValueNNAgent

class NNVDataGeneratorService:
    def __init__(self, logger):
        self.logger = logger
    
    def generate_exploratory_data(self, data_dir, attempt, input_dim, difficulty, config):
        self.logger.log(f"🤖 Cycle {attempt}.1: Generating new exploratory data...")
        output_path = os.path.join(data_dir, f"data_exploratory_attempt_{attempt}.pt")
        generate_value_nn_data(
            agent_class=RandomAgent, model_path=None, input_dim=input_dim,
            difficulties=[difficulty], num_games=config['value_nn_curriculum_config']['exploratory_games'],
            config=config, output_path=output_path
        )

    def generate_agent_data(self, data_dir, attempt, model_path, input_dim, difficulties, config):
        self.logger.log(f"🤖 Cycle {attempt}.1: Generating new agent data...")
        output_path = os.path.join(data_dir, f"data_agent_attempt_{attempt}.pt")
        generate_value_nn_data(
            agent_class=ValueNNAgent, model_path=model_path, input_dim=input_dim,
            difficulties=difficulties, num_games=config['value_nn_curriculum_config']['games_per_generation'],
            config=config, output_path=output_path
        )