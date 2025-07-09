from game.pandemic_game_extended import PandemicGameExtended

class GameManager:
    def get_feature_vector_size(self, config, difficulty):
        temp_env = PandemicGameExtended(difficulty=difficulty, config=config)
        size = temp_env.get_feature_vector_size()
        del temp_env
        return size