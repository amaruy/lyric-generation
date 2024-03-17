import torch
import torch.nn as nn
from torch.utils.data import Dataset



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
