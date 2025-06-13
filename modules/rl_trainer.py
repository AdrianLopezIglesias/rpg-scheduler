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
    log(f"Training on '{rl_cfg['difficulty']}' map.")
    
    env = PandemicGame(difficulty=rl_cfg['difficulty'], config=config)
    
    input_dim = 2 # [cubes, is_player_here]
    agent = GNNAgent(input_dim=input_dim, config=config) # output_dim removed
    
    if rl_cfg.get("load_model_path"):
        try:
            log(f"Loading pre-trained model from: {rl_cfg['load_model_path']}")
            agent.load_model(rl_cfg['load_model_path'])
        except FileNotFoundError:
            log(f"Warning: Pre-trained model not found at {rl_cfg['load_model_path']}. Starting from scratch.")
 

    scores_deque = deque(maxlen=100)
    
    for i_episode in range(1, rl_cfg['num_episodes'] + 1):
        state = env.reset()
        
        for t in range(env.max_actions_per_game):
            # Pass the whole env (`game`) to `choose_action`
            state.batch = torch.zeros(state.num_nodes, dtype=torch.long)
            action = agent.choose_action(env, state) 
            
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