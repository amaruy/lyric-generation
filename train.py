import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm.auto import tqdm
import pickle
from torch.utils.data import Dataset, DataLoader
import time
import sys

from torch.utils.tensorboard import SummaryWriter

class LSTMModel(nn.Module):
    def __init__(self, embedding_weights, hidden_dim, vocab_size, num_layers=1):
        """
        Initializes the LSTM model.
        
        Parameters:
        - embedding_weights: Pre-trained Word2Vec embeddings.
        - hidden_dim: The number of features in the hidden state `h` of the LSTM.
        - vocab_size: The size of the vocabulary.
        - num_layers: Number of recurrent layers (default=1).
        
        The input to the model is expected to be a batch of word indices,
        and the output is a batch of predictions for the next word.
        """
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding layer with pre-trained weights
        self.word_embeddings = nn.Embedding.from_pretrained(embedding_weights, freeze=True)

        # The LSTM takes word embeddings as inputs and outputs hidden states
        self.lstm = nn.LSTM(embedding_weights.shape[1], hidden_dim, num_layers, batch_first=True)

        # The linear layer maps from hidden state space to vocabulary space
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_word_indices):
        """
        Defines the forward pass of the model.
        
        Parameters:
        - input_word_indices: A batch of word indices as input.
        
        Returns:
        - output: The model's predictions for the next word.
        """
        embeddings = self.word_embeddings(input_word_indices)
        lstm_out, _ = self.lstm(embeddings)
        output = self.linear(lstm_out)
        return output
    
class LyricsDataset(Dataset):
    def __init__(self, lyrics_dict):
        """
        Initializes the dataset with preprocessed lyrics.
        
        Parameters:
        lyrics_dict (dict): A dictionary where keys are 'song_name artist' and values are lists of word indices.
        """
        self.lyrics_indices = [indices for indices in lyrics_dict.values()]
        self.all_indices = [idx for sublist in self.lyrics_indices for idx in sublist]
    
    def __len__(self):
        """Returns the total number of word indices in the dataset."""
        return len(self.all_indices) - 1  # Subtract 1 because we use a look-ahead of 1 for targets
    
    def __getitem__(self, index):
        """
        Returns a tuple (current_word_index, next_word_index) for training.
        
        Parameters:
        index (int): The index of the current word.
        
        Returns:
        tuple: A tuple of tensors (current_word_index, next_word_index).
        """
        return (torch.tensor(self.all_indices[index], dtype=torch.long), 
                torch.tensor(self.all_indices[index + 1], dtype=torch.long))
    
def train_model(model, train_loader, config):
    """
    Trains the LSTM model on the given dataset.
    
    Parameters:
    - model: The LSTM model to train.
    - train_loader: DataLoader for the training dataset.
    - config: Dictionary containing configuration parameters.
    """
    writer = SummaryWriter(f"runs/{config['experiment_name']}")
    model.train()  # Switch model to training mode
    optimizer = Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    model.to(config["device"])
    
    for epoch in range(config["epochs"]):
        total_loss = 0
        verbosity = False
        # only show progress bar if verbosity is True
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), leave=False, disable=not verbosity)
        for i, (input_words, target_words) in progress_bar:
            input_words, target_words = input_words.to(config["device"]), target_words.to(config["device"])
            
            # Forward pass
            output = model(input_words)
            loss = criterion(output.view(-1, config["vocab_size"]), target_words.view(-1))
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_description(f"Epoch {epoch+1} Loss: {total_loss/(i+1):.4f}")
            
        # Log the average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('training_loss', avg_loss, epoch+1)
        
        print(f"Epoch {epoch+1} Completed. Avg Loss: {avg_loss:.4f}")

def memory_stats():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print(f'total memory: {t}, reserved memory: {r}, allocated memory: {a}, free memory: {f}')

print("Starting training...", flush=True)
# Configuration parameters
config = {
    "batch_size": 8,
    "learning_rate": 0.001,
    "hidden_dim": 128,
    "num_layers": 2,
    "embedding_dim": 300,  # Assuming we know the embedding dimension
    "vocab_size": 3000000,  # Make sure this matches the actual vocab size
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "shuffle": False,
    "epochs": 10
}

# clear cached memory if config['device'] is cuda
if config['device'] == torch.device('cuda'):
    torch.cuda.empty_cache()


config["experiment_name"] = f"lstm_{config['hidden_dim']}_{config['num_layers']}_{config['learning_rate']}"
print(config)
print("Loading pre-trained Word2Vec embeddings...")
# load pre-trained Word2Vec embeddings
embedding_weights = torch.load('models/word2vec_weights.pt')
vocab_size, embedding_dim = embedding_weights.shape
device = config['device']

# Initialize the LSTM model with Word2Vec weights and corrected vocabulary size
lstm_model = LSTMModel(embedding_weights=embedding_weights, hidden_dim=256, vocab_size=vocab_size, num_layers=1)
lstm_model.to(device)
print("LSTM Model:")
print(lstm_model)

# print memory stats
memory_stats()


# Load the pre-processed dataset
with open('data/lyrics_dict.pkl', 'rb') as f:
    lyrics_dict = pickle.load(f)

dataset = LyricsDataset(lyrics_dict)
train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=config['shuffle'])
print("Dataset loaded successfully!")
print(f"Number of training samples: {len(dataset)}")
print('Training...')
start = time.time()
# train the model
train_model(lstm_model, train_loader, config)
print(f"Training complete! Time taken: {time.time() - start:.2f} seconds")
print(f"saving model weights to models/{config['experiment_name']}_weights.pth...")
# save model weights
torch.save(lstm_model.state_dict(), f'models/{config["experiment_name"]}_weights.pth')

print("Done!")