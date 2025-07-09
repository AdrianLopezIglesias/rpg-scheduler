class CurriculumOrchestrator:
    def __init__(self, logger, config, data_repo, trainer, data_gen, validator, game_manager):
        self.logger = logger
        self.config = config
        self.curriculum_cfg = config['value_nn_curriculum_config']
        self.data_repo = data_repo
        self.trainer = trainer
        self.data_gen = data_gen
        self.validator = validator
        self.game_manager = game_manager

    def run(self):
        self.data_repo.setup_directories("data/value_nn_run_data/", "models/")
        
        input_dim = self.game_manager.get_feature_vector_size(
            self.config, self.curriculum_cfg['difficulties'][0]
        )
        
        last_successful_model = None
        active_difficulties = []

        for difficulty in self.curriculum_cfg['difficulties']:
            active_difficulties.append(difficulty)
            self.logger.log_stage_start(difficulty, active_difficulties)
            self.data_repo.clear_directory("data/value_nn_run_data/")
            
            stage_passed, last_successful_model = self._run_stage(
                difficulty, active_difficulties, input_dim, last_successful_model
            )
            
            if not stage_passed:
                self.logger.log_curriculum_failed(difficulty)
                return

        self.logger.log_curriculum_success()

    def _run_stage(self, difficulty, active_difficulties, input_dim, model_to_load):
        for i in range(self.curriculum_cfg['max_retries']):
            attempt = i + 1
            self.logger.log_cycle_start(attempt, self.curriculum_cfg['max_retries'], difficulty)
            
            self.data_gen.generate_exploratory_data(
                "data/value_nn_run_data/", attempt, input_dim, difficulty, self.config
            )

            candidate_model = self.trainer.train_candidate_model(
                input_dim, "data/value_nn_run_data/", model_to_load,
                self.config, difficulty, attempt
            )

            self.data_gen.generate_agent_data(
                "data/value_nn_run_data/", attempt, candidate_model, input_dim,
                active_difficulties, self.config
            )

            val_results = self.validator.validate(self.config, candidate_model, difficulty)
            
            win_rate = val_results.get("win_rate_percent", 0)
            target_rate = self.curriculum_cfg['targets'][difficulty]['target_win_rate']
            
            is_met = win_rate >= target_rate
            self.logger.log_validation_check(attempt, target_rate, win_rate, is_met)
            
            if is_met:
                self.logger.log_stage_success(difficulty)
                return True, candidate_model
            else:
                model_to_load = candidate_model
                
        return False, None