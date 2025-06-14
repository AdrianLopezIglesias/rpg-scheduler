import torch
from torch.distributions import Categorical
import numpy as np
from game.pandemic_game import PandemicGame
# --- UPDATED IMPORT ---
from agents import GNNAgent
from modules.utils import log
import textwrap

def run_single_train_step_debug(config):
    debug_cfg = config['debug_train_config']
    log("=============== DEBUGGING SINGLE TRAINING STEP ===============")
    log(f"Playing on '{debug_cfg['difficulty']}' map.")

    env = PandemicGame(difficulty=debug_cfg['difficulty'], config=config)
    input_dim = env.get_node_feature_count()
    agent = GNNAgent(input_dim=input_dim, config=config)
    log("Initialized new, untrained agent and environment.")

    state = env.reset()
    done = False
    turn = 0
    while not done:
        turn += 1
        log(f"\n{'='*15} Turn {turn} {'='*15}")

        loc = env.player_location
        cubes_data = {city: data['cubes'] for city, data in env.board_state.items()}
        non_zero_cubes = {city: {c:v for c,v in cubes.items() if v > 0} for city, cubes in cubes_data.items()}
        non_zero_cubes = {city: cubes for city, cubes in non_zero_cubes.items() if cubes}
        log(f"Game State: Player at '{loc}', Cubes at {non_zero_cubes}")

        state.batch = torch.zeros(state.num_nodes, dtype=torch.long)
        log("Model Input (Graph Data Object):")
        log(f"  - Node Features (x) [Shape: {state.x.shape}]:")
        for i, city in enumerate(env.all_cities):
            log(f"      {city:<15}: {state.x[i].numpy()}")
        edge_str = textwrap.fill(str(state.edge_index.numpy().tolist()), width=70, subsequent_indent='     ')
        log(f"  - Edge Index (edge_index) [Shape: {state.edge_index.shape}]:\n     {edge_str}")

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
                elif action_desc.get('type') == 'discover_cure':
                    color_scores = agent.policy_network.cure_head(graph_embedding)
                    color_idx = agent.colors.index(action_desc['color'])
                    score = color_scores[0, color_idx]
                elif action_desc['type'] == 'pass':
                    score = agent.policy_network.pass_head(graph_embedding)
                logits[action_idx] = score
                log(f"    - Legal Action: {action_desc}, Raw Score: {score.item():.4f}")

        prob_dist = Categorical(logits=logits)
        chosen_action_idx = prob_dist.sample()
        chosen_action_desc = env.idx_to_action[chosen_action_idx.item()]
        log(f"==> Agent chose: {chosen_action_desc}")
        
        state, reward, done = env.step(chosen_action_idx.item())
        agent.rewards.append(reward)
        agent.log_probs.append(prob_dist.log_prob(chosen_action_idx))

    log(f"\n{'='*15} EPISODE END {'='*15}")
    final_result = env.is_game_over()[1]
    log(f"Game Over. Final Result: {final_result}")

    log("\n--- Agent Learning Calculation ---")
    log(f"Raw Rewards collected this episode: {agent.rewards}")

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