import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm.auto import tqdm
import pickle
from torch.utils.data import Dataset, DataLoader
import time
import sys

from torch.utils.tensorboard import SummaryWriter


print("Starting training...", flush=True)
# Configuration parameters
config = {
    "batch_size": 2,
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

# load weights if available
try:
    lstm_model.load_state_dict(torch.load(f'models/{config["experiment_name"]}_weights.pth'))
    print(f"Loaded model weights from models/{config['experiment_name']}_weights.pth")
    config['experiment_name'] += "_continued"
except FileNotFoundError:
    print(f"No model weights found at models/{config['experiment_name']}_weights.pth")

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