from .value_nn_trainer import train_value_nn
from agents.value_nn_agent import ValueNet

class NNVTrainerService:
    def __init__(self, logger, data_repo, model_repo):
        self.logger = logger
        self.data_repo = data_repo
        self.model_repo = model_repo

    def train_candidate_model(self, input_dim, data_dir, model_to_load, config, difficulty, attempt):
        self.logger.log(f"--- 🧠 Attempt {attempt}: Training Phase ---")
        
        aggregated_data = self.data_repo.load_all(data_dir)
        self.logger.log(f"    📚 Total training samples: {len(aggregated_data)}")
        
        if not aggregated_data:
            self.logger.log("    ⚠️ No training data. Skipping training.")
            return model_to_load
            
        model = ValueNet(input_dim)
        self.model_repo.load_into(model, model_to_load)

        train_value_nn(model, aggregated_data, config)
        
        candidate_path = f"models/value_nn_diff_{difficulty}_attempt_{attempt}.pth"
        self.model_repo.save(model.state_dict(), candidate_path)
        return candidate_path