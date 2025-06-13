import torch
from collections import deque
import numpy as np
from game.pandemic_game import PandemicGame
from agents.agents import GNNAgent
from .utils import log

def run_rl_training(config):
    """The main RL training loop."""
    rl_cfg = config['rl_config']
    log("=============== STARTING RL TRAINING RUN ===============")
    
    env = PandemicGame(difficulty=rl_cfg['difficulty'], config=config)
    
    # Node features: [cubes, is_player_here] -> 2
    input_dim = 2
    output_dim = env.get_action_space_size()
    agent = GNNAgent(input_dim=input_dim, output_dim=output_dim, config=config)
    
    scores_deque = deque(maxlen=100)
    
    for i_episode in range(1, rl_cfg['num_episodes'] + 1):
        state = env.reset()
        
        for t in range(env.max_actions_per_game):
            action_mask = env.get_possible_action_mask()
            action = agent.choose_action(state, action_mask)
            
            next_state, reward, done = env.step(action)
            agent.rewards.append(reward)
            
            if done:
                break
            state = next_state
        
        scores_deque.append(sum(agent.rewards))
        agent.update_policy()
        
        if i_episode % rl_cfg['log_interval'] == 0:
            avg_score = np.mean(scores_deque)
            log(f'Episode {i_episode}\tAverage Score (last 100): {avg_score:.2f}')
    
    log(f"\nTraining finished. Saving model to {rl_cfg['model_save_path']}")
    agent.save_model(rl_cfg['model_save_path'])
