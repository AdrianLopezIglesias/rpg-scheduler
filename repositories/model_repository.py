import os
import torch
from agents.value_nn_agent import ValueNet

class ModelRepository:
    def __init__(self, logger):
        self.logger = logger

    def save(self, model_state, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model_state, path)
        self.logger.log(f"    💾 Saved candidate model to '{path}'")

    def load_into(self, model, model_path):
        if not model_path:
            return
        try:
            model.load_state_dict(torch.load(model_path))
            self.logger.log(f"    ☑️  Loaded weights from '{os.path.basename(model_path)}' into new model.")
        except Exception as e:
            self.logger.log(f"    ⚠️  Could not load model weights, starting fresh. Error: {e}")