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

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, num_layers=1):
        """
        Initializes the modified LSTM model to accept concatenated word and MIDI embeddings as input.

        Parameters:
        - input_dim (int): The dimensionality of the concatenated input vector (word embedding + MIDI embedding).
        - hidden_dim (int): The number of features in the hidden state `h` of the LSTM.
        - vocab_size (int): The size of the vocabulary, used for the output layer dimension.
        - num_layers (int): Number of recurrent layers (default=1).
        
        The input to the model is expected to be a batch of concatenated word and MIDI embeddings,
        and the output is a batch of predictions for the next word.
        """
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layer takes concatenated embeddings as inputs
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Linear layer that maps from hidden state space to vocabulary space
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, concatenated_embeddings):
        """
        Defines the forward pass of the model using concatenated word and MIDI embeddings.
        
        Parameters:
        - concatenated_embeddings: A batch of concatenated word and MIDI embeddings.
        
        Returns:
        - output: The model's predictions for the next word.
        """
        lstm_out, _ = self.lstm(concatenated_embeddings)
        output = self.linear(lstm_out)
        return output

def train_model(model, train_loader, config):
    """
    Trains the LSTM model on the given dataset, using preprocessed concatenated embeddings.

    Parameters:
    - model (torch.nn.Module): The LSTM model to be trained.
    - train_loader (DataLoader): DataLoader for the training dataset, providing batches of preprocessed inputs and targets.
    - config (dict): Configuration parameters including epochs, learning rate, device, etc.

    The DataLoader is expected to yield batches of (input_features, target_words), where:
    - input_features are the concatenated word and MIDI embeddings,
    - target_words are the indices of the target words to predict.
    """
    writer = SummaryWriter(f"runs/{config['experiment_name']}")
    model.train()  # Ensure the model is in training mode
    optimizer = Adam(model.parameters(), lr=config["learning_rate"])
    criterion = CrossEntropyLoss()  # Appropriate for classification tasks
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


def memory_stats():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print(f'total memory: {t}, reserved memory: {r}, allocated memory: {a}, free memory: {f}')


class LyricsMIDIDataset(Dataset):
    def __init__(self, lyrics_dict=None, midi_embeddings=None, word2vec_model=None, preloaded_inputs=None, preloaded_targets=None):
        """
        Initializes the dataset with either raw data to be processed or preloaded processed data.
        
        Parameters:
        - lyrics_dict (dict): Raw lyrics data.
        - midi_embeddings (dict): Raw MIDI embeddings.
        - word2vec_model: The Word2Vec model.
        - preloaded_inputs (torch.Tensor): Preloaded inputs tensor.
        - preloaded_targets (torch.Tensor): Preloaded targets tensor.
        """
        if preloaded_inputs is not None and preloaded_targets is not None:
            self.inputs = preloaded_inputs
            self.targets = preloaded_targets
        else:
            self.inputs = []
            self.targets = []
            # get all songs that arent in keys of midi_embeddings or lyrics_dict
            self.missing_songs = set(lyrics_dict.keys()) ^ set(midi_embeddings.keys())

            for song_key, word_indices in lyrics_dict.items():
                if song_key not in midi_embeddings:
                    continue  # Skip songs without a corresponding MIDI embedding
                midi_embedding = midi_embeddings[song_key]  # MIDI embedding for the current song

                for i in range(len(word_indices) - 1):
                    # Convert word indices to embeddings
                    word_embedding_current = word2vec_model[word_indices[i]]
                    next_word_indice = word_indices[i + 1]

                    # Concatenate word embedding with MIDI embedding for the input
                    word_embedding_current = word_embedding_current.reshape(1, -1)  # Reshape to (1, embedding_dim)
                    input_feature = np.concatenate([word_embedding_current, midi_embedding], axis=1)
                    self.inputs.append(input_feature)
                    self.targets.append(next_word_indice)

            # Convert lists to tensors for PyTorch compatibility
            self.inputs = torch.tensor(self.inputs, dtype=torch.float)
            self.targets = torch.tensor(self.targets, dtype=torch.long)

    def __len__(self):
        """Returns the total number of input-target pairs."""
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Returns an input-target pair by index.
        
        Parameters:
        - idx (int): The index of the input-target pair.
        
        Returns:
        - tuple: A tuple containing the input feature tensor and target tensor.
        """
        return self.inputs[idx], self.targets[idx]
    
config = {
    "batch_size": 64,
    "learning_rate": 0.001,
    "hidden_dim": 128,
    "num_layers": 2,
    "embedding_dim": 300,  # Assuming we know the embedding dimension
    "vocab_size": 3000000,  # Make sure this matches the actual vocab size
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "shuffle": False,
    "epochs": 100,
    'verbosity': False
}
print("Starting training...", flush=True)
print(config)


# clear cached memory if config['device'] is cuda
if config['device'] == torch.device('cuda'):
    torch.cuda.empty_cache()

print('Loading dataset...')
# Load the dataset with preloaded inputs and targets
config["experiment_name"] = f"lstm_int_{config['hidden_dim']}_{config['num_layers']}_{config['learning_rate']}"
preloaded_inputs = torch.load('data/dataset_saved_inputs.pt')
preloaded_targets = torch.load('data/dataset_saved_targets.pt')
preloaded_targets = torch.tensor(preloaded_targets, dtype=torch.long)  # Ensure targets are long
dataset = LyricsMIDIDataset(preloaded_inputs=preloaded_inputs, preloaded_targets=preloaded_targets)

train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=config['shuffle'])

input_dim = dataset.inputs.shape[2]
print("Input Dimension:", input_dim)
print("Dataset loaded successfully!")
print(f"Number of training samples: {len(dataset)}")
print('Initializing LSTM model...')
lstm_model = LSTMModel(input_dim=input_dim, hidden_dim=config['hidden_dim'], vocab_size=config['vocab_size'], num_layers=config['num_layers'])
print("Training...", flush=True)
start = time.time()
train_model(lstm_model, train_loader, config)
print(f"Training complete! Time taken: {time.time() - start:.2f} seconds")
print(f"saving model weights to models/{config['experiment_name']}_weights.pth...")
# save model weights
torch.save(lstm_model.state_dict(), f'models/{config["experiment_name"]}_weights.pth')

print("Done!")
