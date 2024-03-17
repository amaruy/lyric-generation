import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np


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





