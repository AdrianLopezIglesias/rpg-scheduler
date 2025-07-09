import torch
import torch.optim as optim
import torch.nn.functional as F
import random
from agents.value_nn_agent import ValueNet

def train_value_nn(model, training_data, config, device='cpu'):
    """
    Trains the ValueNet model in a supervised manner.
    """
    critic_cfg = config.get("critic_trainer_config", {})
    epochs = critic_cfg.get("epochs", 3)
    batch_size = critic_cfg.get("batch_size", 128)
    learning_rate = float(config.get('value_nn_curriculum_config', {}).get('learning_rate', 0.001))

    if not training_data:
        print("--- ValueNet training skipped: No data provided. ---")
        return

    print(f"--- Starting ValueNet training: {len(training_data)} examples, {epochs} epochs, batch size {batch_size} ---")

    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        random.shuffle(training_data)
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, len(training_data), batch_size):
            batch_data = training_data[i:i+batch_size]
            states, targets = zip(*batch_data)

            state_tensors = torch.tensor(states, dtype=torch.float32).to(device)
            target_tensors = torch.stack(list(targets)).to(device)

            optimizer.zero_grad()
            predicted_values = model(state_tensors)
            loss = F.mse_loss(predicted_values, target_tensors)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"  Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

    print("--- ValueNet training finished. ---")
