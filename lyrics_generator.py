import torch
import torch.nn.functional as F
import pickle
import gensim
from gensim.models import KeyedVectors
import numpy as np
from models.lstm import LSTMModel

class LyricsGenerator:
    def __init__(self, lstm_model, word2vec_model, midi_embeddings=None, device='cpu'):
        """
        Initializes the LyricsGenerator with the necessary models and embeddings.
        
        Parameters:
        - lstm_model: The trained LSTMMidiModel instance.
        - word2vec_model: A pre-loaded Word2Vec model used for converting words to embeddings.
        - midi_embeddings: A dictionary of MIDI embeddings keyed by song.
        - device (str): The device to run the model on ('cpu' or 'cuda').
        """
        self.lstm_model = lstm_model.to(device)
        self.word2vec_model = word2vec_model
        self.midi_embeddings = midi_embeddings
        self.device = device

    def _sample_next_word(self, logits, temperature=1.0):
        """
        Samples the next word from the logits with a given temperature.

        Parameters:
        - logits: The logits output by the model.
        - temperature: Controls the randomness of the sampling. Higher values lead to more random outputs.

        Returns:
        - index of the sampled word.
        """
        if temperature <= 0:
            return torch.argmax(logits, dim=-1).item()
        probabilities = F.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probabilities, 1).item()

    def generate(self, song_key=None, seed_text='BOS', max_length=50, temperature=1.0):
        """
        Generates lyrics based on a seed text and a song key for MIDI context.
        
        Parameters:
        - seed_text (str): The initial text to start generating from, 'BOS' if no input is given.
        - song_key (str): The key identifying the song's MIDI embedding to use.
        - max_length (int): Maximum length of the generated lyrics.
        - temperature (float): Controls the randomness of the sampling. Higher values lead to more random outputs.
        
        Returns:
        - str: The generated lyrics.
        """
        self.lstm_model.eval()
        generated_text = [seed_text]
        current_word = seed_text
        
        # Ensure the song key is valid
        if song_key is None: 
            input_size = self.lstm_model.lstm.input_size
            midi_embedding = torch.zeros(1, input_size - self.word2vec_model.vector_size, device=self.device)
        elif song_key not in self.midi_embeddings:
            raise ValueError("Song key not found in MIDI embeddings.") 
        else:
            midi_embedding = torch.tensor(self.midi_embeddings[song_key], dtype=torch.float, device=self.device).squeeze(0)
        
        # Start generating words one by one
        for _ in range(max_length):
            current_embeddings = torch.tensor(self.word2vec_model[current_word], dtype=torch.float, device=self.device).squeeze(0)
            # Assuming midi_embedding is ready to use and matches input dimensions
            input_feature = torch.cat((current_embeddings, midi_embedding), dim=-1)
            
            # Generate the next word
            with torch.no_grad():

                output = self.lstm_model(input_feature.unsqueeze(0))
                # Convert output to word
                next_word_index = self._sample_next_word(output, temperature)
                next_word = self.word2vec_model.index_to_key[next_word_index]
                if next_word == 'EOF':
                    break
                generated_text.append(next_word)
                current_word = next_word


        # clean generated text, string, replace EOF, EOS, BOS
        generated_text = ' '.join(generated_text[1:]).replace('EOF', '').replace('EOS', '\n').replace(' BOS', '')
        return generated_text


def main():
    # load midi_embeddings.pkl
    with open('data/midi_embeddings.pkl', 'rb') as f:
        midi_embeddings = pickle.load(f)

    # load word2vec model
    word2vec_model = KeyedVectors.load('models/weights/word2vec-google-news-300.model')

    # Load the trained LSTM model
    input_size = word2vec_model.vector_size + len(list(midi_embeddings.values())[0][0])
    lstm_model = LSTMModel(input_dim=input_size, hidden_dim=128, vocab_size=len(word2vec_model), num_layers=2)
    lstm_model.load_state_dict(torch.load('models/weights/lstmmidi_128_2_0.001_weights.pth'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lyrics_generator = LyricsGenerator(lstm_model, word2vec_model, midi_embeddings, device=device)
    print("Lyrics Generator Initialized!")
    # Generate lyrics with a specific song key and seed text
    song_key = input("Enter a song_name artist lke 'hello adele' to generate lyrics:")
    seed_text = 'BOS'
    max_length = 100
    temperature = 1.0  # Adjust for creativity


    while song_key:
        generated_lyrics = lyrics_generator.generate(song_key=song_key, seed_text=seed_text, max_length=max_length, temperature=temperature)
        print(f"Generated Lyrics based on the music of {song_key}:\n", generated_lyrics)
        song_key = input("Enter a song_name artist lke 'hello adele' to generate lyrics:")


    
if __name__ == "__main__":
    main()

    