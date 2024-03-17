import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time

def memory_stats():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print(f'total memory: {t}, reserved memory: {r}, allocated memory: {a}, free memory: {f}')
 

def clear_memory(device):
    if device == torch.device('cuda'):
        torch.cuda.empty_cache()

def load_pretrained(model, weight_path):
    try:
        model.load_state_dict(torch.load(weight_path))
        print(f"Loaded model weights from {weight_path}")
    except FileNotFoundError:
        print(f"No model weights found at {weight_path}")
    return model

def train_model(model, train_set, config):
    """
    Trains the LSTM model on the given dataset, using preprocessed concatenated embeddings.

    Parameters:
    - model (torch.nn.Module): The LSTM model to be trained.
    - train_set (Dataset): training dataset, preprocessed inputs and targets.
    - config (dict): Configuration parameters including epochs, learning rate, device, etc.

    The Dataset is expected to yield (input_features, target_words), where:
    - input_features are the concatenated word and MIDI embeddings,
    - target_words are the indices of the target words to predict.
    """
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=config['shuffle'])  
    writer = SummaryWriter(f"runs/{config['experiment_name']}")
    model.train()  # Ensure the model is in training mode
    optimizer = Adam(model.parameters(), lr=config["learning_rate"])
    criterion = CrossEntropyLoss() 
    model.to(config["device"])  # Move model to configured device (CPU/GPU)

    for epoch in range(config["epochs"]):
        total_loss = 0.0
        # Enable or disable the progress bar based on verbosity setting
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), leave=False, disable=not config.get("verbosity", False))
        
        for i, (input_features, target_words) in progress_bar:
            input_features, target_words = input_features.to(config["device"]), target_words.to(config["device"])
            
            optimizer.zero_grad()  # Clear gradients
            outputs = model(input_features)  # Forward pass
            loss = criterion(outputs.view(-1, config["vocab_size"]), target_words.view(-1))  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            total_loss += loss.item()  # Accumulate loss
            progress_bar.set_description(f"Epoch {epoch+1} Loss: {total_loss/(i+1):.4f}")
        
        avg_loss = total_loss / len(train_loader)  # Calculate average loss
        writer.add_scalar('training_loss', avg_loss, epoch+1)  # Log to TensorBoard
        print(f"Epoch {epoch+1} Completed. Avg Loss: {avg_loss:.4f}")

    writer.close()  # Close the TensorBoard writer
    # Save the model weights
    if config.get("save", True):
        torch.save(model.state_dict(), f'models/{config["experiment_name"]}_weights.pth')
        print(f"Model weights saved to models/{config['experiment_name']}_weights.pth")
    return model