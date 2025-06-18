# modules/critic_trainer.py

import random
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

def train_critic_from_data(agent, training_data, config,difficulty):
    """
    Trains the Critic (value_head) of the PolicyNetwork in a supervised manner.
    It uses a dataset of (state, outcome) pairs to teach the Critic to predict
    the final value of a given game state.

    Args:
        agent: The agent object, which holds the policy_network and optimizer.
        training_data (list): The dataset of (state_graph, target_value) tuples.
        config (dict): The main configuration object.
    """
    # Get hyperparameters from config. You can create this new section.
    critic_cfg = config.get("critic_trainer_config", {})
    # epochs = critic_cfg.get("epochs", 3) * difficulty
    epochs = critic_cfg.get("epochs", 3) 
    batch_size = critic_cfg.get("batch_size", 128)
    
    if not training_data:
        print("--- Critic training skipped: No data provided. ---")
        return

    print(f"--- Starting Critic training: {len(training_data)} examples, {epochs} epochs, batch size {batch_size} ---")

    # Set the network to training mode
    agent.policy_network.train()
    
    for epoch in range(epochs):
        # Shuffle data before each epoch for better training
        random.shuffle(training_data)
        epoch_loss = 0.0
        num_batches = 0

        # Iterate through the data in mini-batches
        for i in range(0, len(training_data), batch_size):
            batch_data = training_data[i:i+batch_size]
            states, targets = zip(*batch_data)
            
            # Use torch_geometric's Batch object to process multiple graphs at once
            state_batch = Batch.from_data_list(list(states))
            target_batch = torch.stack(list(targets)).squeeze()

            # Zero the gradients from the previous step
            agent.optimizer.zero_grad()
            
            # Get the Critic's prediction for the entire batch of states
            # We only need the third value returned, the predicted state values
            _, _, predicted_values = agent.policy_network(state_batch)
            
            # Calculate the loss. Mean Squared Error is ideal for this regression task.
            loss = F.mse_loss(predicted_values.squeeze(), target_batch)
            
            # Backpropagate the loss through the network
            loss.backward()
            
            # Update the network's weights
            agent.optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"  Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
    print("--- Critic training finished. ---")
    # The agent's model has been updated in place, so we don't need to return anything.