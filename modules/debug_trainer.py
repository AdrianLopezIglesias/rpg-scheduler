import torch
from torch.distributions import Categorical
import numpy as np
from game.pandemic_game import PandemicGame
from agents.agents import GNNAgent
from modules.utils import log
import textwrap

def run_single_train_step_debug(config):
    """
    Runs a single, highly-verbose training episode to show how the agent learns.
    Does not update or save the model.
    """
    debug_cfg = config['debug_train_config']
    log("=============== DEBUGGING SINGLE TRAINING STEP ===============")
    log(f"Playing on '{debug_cfg['difficulty']}' map.")

    # 1. Initialization
    env = PandemicGame(difficulty=debug_cfg['difficulty'], config=config)
    input_dim = 2
    agent = GNNAgent(input_dim=input_dim, config=config)
    log("Initialized new, untrained agent and environment.")

    # 2. Play one full game (episode)
    state = env.reset()
    done = False
    turn = 0
    while not done:
        turn += 1
        log(f"\n{'='*15} Turn {turn} {'='*15}")

        # --- Log Game State ---
        loc = env.player_location
        cubes = {city: data['cubes'] for city, data in env.board_state.items() if data['cubes'] > 0}
        log(f"Game State: Player at '{loc}', Cubes at {cubes}")

        # --- Log Model Input ---
        state.batch = torch.zeros(state.num_nodes, dtype=torch.long)
        log("Model Input (Graph Data Object):")
        log(f"  - Node Features (x) [Shape: {state.x.shape}]:")
        for i, city in enumerate(env.all_cities):
            log(f"      {city:<15}: {state.x[i].numpy()}")
        edge_str = textwrap.fill(str(state.edge_index.numpy().tolist()), width=70, subsequent_indent='     ')
        log(f"  - Edge Index (edge_index) [Shape: {state.edge_index.shape}]:\n     {edge_str}")

        # --- Log Model Evaluation ---
        log("Model Evaluation:")
        (node_embeddings, graph_embedding) = agent.policy_network(state)
        possible_actions_mask = env.get_possible_action_mask()
        logits = torch.full_like(possible_actions_mask, -1e8, dtype=torch.float)
        
        log("  Action Scores (Logits):")
        for action_idx, is_possible in enumerate(possible_actions_mask):
            if is_possible:
                action_desc = env.idx_to_action[action_idx]
                score = 0
                if action_desc['type'] == 'move':
                    score = agent.policy_network.move_head(node_embeddings[action_desc['target_idx']])
                elif action_desc['type'] == 'treat':
                    score = agent.policy_network.treat_head(node_embeddings[action_desc['target_idx']])
                elif action_desc['type'] == 'pass':
                    score = agent.policy_network.pass_head(graph_embedding)
                logits[action_idx] = score
                log(f"    - Legal Action: {action_desc}, Raw Score: {score.item():.4f}")

        # --- Log Action Selection ---
        prob_dist = Categorical(logits=logits)
        chosen_action_idx = prob_dist.sample()
        chosen_action_desc = env.idx_to_action[chosen_action_idx.item()]
        log(f"==> Agent chose: {chosen_action_desc}")
        
        # Take step and store results for learning
        state, reward, done = env.step(chosen_action_idx.item())
        agent.rewards.append(reward)
        agent.log_probs.append(prob_dist.log_prob(chosen_action_idx))

    log(f"\n{'='*15} EPISODE END {'='*15}")
    final_result = env.is_game_over()[1]
    log(f"Game Over. Final Result: {final_result}")

    # 3. Log Learning Calculation
    log("\n--- Agent Learning Calculation ---")
    log(f"Raw Rewards collected this episode: {agent.rewards}")

    # Calculate discounted rewards
    discounted_rewards = []
    R = 0
    for r in agent.rewards[::-1]:
        R = r + agent.gamma * R
        discounted_rewards.insert(0, R)
    log(f"Discounted Rewards (Credit/Blame): {np.round(discounted_rewards, 2).tolist()}")

    if len(discounted_rewards) > 1:
        rewards_tensor = torch.tensor(discounted_rewards, dtype=torch.float32)
        normalized_rewards = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-9)
        log(f"Normalized Rewards (For stable training): {np.round(normalized_rewards.numpy(), 2).tolist()}")
    else:
        normalized_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)

    # Calculate policy loss
    policy_loss = []
    log("Calculating Policy Loss (multiplying log probability of action by its final credit/blame):")
    for i, (log_prob, reward) in enumerate(zip(agent.log_probs, normalized_rewards)):
        loss_term = -log_prob * reward
        policy_loss.append(loss_term)
        log(f"  - Step {i+1}: -log_prob({log_prob.item():.4f}) * reward({reward.item():.4f}) = loss_term({loss_term.item():.4f})")
    
    final_loss = torch.stack(policy_loss).sum()
    log(f"Total Loss for this episode: {final_loss.item():.4f}")
    log("This loss would then be used to update the model's weights.")
    log("\n=============== DEBUGGING FINISHED ===============")