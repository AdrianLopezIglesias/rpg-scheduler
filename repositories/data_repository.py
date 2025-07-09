import os
import glob
import torch
import shutil

class DataRepository:
    def __init__(self, logger):
        self.logger = logger

    def load_all(self, data_dir):
        all_data = []
        all_files = glob.glob(os.path.join(data_dir, "*.pt"))
        if not all_files:
            return all_data
        
        self.logger.log(f"🗂️  Loading {len(all_files)} data files...")
        for f_path in all_files:
            try:
                all_data.extend(torch.load(f_path, weights_only=False))
            except (IOError, EOFError) as e:
                self.logger.log(f"⚠️  Warning: Could not load {f_path}. Skipping. Error: {e}")
        return all_data

    def setup_directories(self, data_dir, model_dir):
        if os.path.exists(data_dir): shutil.rmtree(data_dir)
        if os.path.exists(model_dir): shutil.rmtree(model_dir)
        os.makedirs(data_dir)
        os.makedirs(model_dir)
        self.logger.log("✅ Created clean data and model directories.")

    def clear_directory(self, data_dir):
        if os.path.exists(data_dir): shutil.rmtree(data_dir)
        os.makedirs(data_dir)
        self.logger.log("   Cleared data directory for new stage.")