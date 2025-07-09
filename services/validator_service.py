from .value_nn_validator import run_value_nn_validation

class NNVValidatorService:
    def __init__(self, logger):
        self.logger = logger
        
    def validate(self, config, model_path, difficulty):
        self.logger.log(f"\n🤖 Validating model performance...")
        return run_value_nn_validation(config, model_path, difficulty)