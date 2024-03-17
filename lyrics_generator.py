import torch
import torch.nn.functional as F
import pickle

class LyricsGenerator:
    def __init__(self, model, vocab, device='cpu'):
        """
        Initializes the LyricsGenerator.

        Parameters:
        - model: The trained LSTM model for lyrics generation.
        - vocab: A mapping from words to indices and indices to words (vocab.stoi and vocab.itos).
        - device: The device to run the generation on ('cpu' or 'cuda').
        """
        self.model = model
        self.vocab = vocab
        self.device = device

    def sample_next_word(self, logits, temperature=1.0):
        """
        Samples the next word from the logits with a given temperature.

        Parameters:
        - logits: The logits output by the model.
        - temperature: Controls the randomness of the sampling. Higher values lead to more random outputs.

        Returns:
        - index of the sampled word.
        """
        probabilities = F.softmax(logits / temperature, dim=-1)
        word_index = torch.multinomial(probabilities, 1).item()
        return word_index

    def generate(self, start_word, max_words=50, max_words_per_line=10, temperature=1.0):
        """
        Generates lyrics starting from a given word.

        Parameters:
        - start_word: The word to start generating from.
        - max_words: The maximum number of words in the generated lyrics.
        - max_words_per_line: The maximum number of words per line.
        - temperature: Controls the randomness of the sampling.

        Returns:
        - A string containing the generated lyrics.
        """
        self.model.eval()  # Set the model to evaluation mode
        words = [start_word]
        current_word_index = torch.tensor([self.vocab.stoi[start_word]], device=self.device)

        for _ in range(max_words - 1):
            with torch.no_grad():
                logits = self.model(current_word_index.unsqueeze(0))[:, -1, :]
                next_word_index = self.sample_next_word(logits, temperature)
                next_word = self.vocab.itos[next_word_index]
                words.append(next_word)
                current_word_index = torch.tensor([next_word_index], device=self.device)

                if len(words) % max_words_per_line == 0:
                    words.append('\n')

            if words[-1] == '<eos>':  # Assuming <eos> is the end-of-sentence token
                break

        return ' '.join(words).replace(' \n ', '\n')


class Vocabulary:
    """
    A mapping from words to indices and indices to words.
    """
    def __init__(self, stoi, itos):
        self.stoi = stoi
        self.itos = itos
    
    def __call__(self, word):
        if word in self.stoi:
            return self.stoi[word]
        else:
            return self.stoi['<unk>']
        
    def __len__(self):
        return len(self.stoi)


# Example usage:
# model = torch.load('lyrics_model.pth')
# vocab = Vocabulary(stoi, itos)
# load stoi and itos
# with open('models/stoi.pkl', 'rb') as f:
#     stoi = pickle.load(f)
# with open('models/itos.pkl', 'rb') as f:
#     itos = pickle.load(f)

# vocab = Vocabulary(stoi, itos)

# # load model
# model = torch.load('models/lyrics_model.pth')
# lyrics_generator = LyricsGenerator(model, vocab, device='cpu')

# # Generate lyrics starting from the word "love"
# lyrics = lyrics_generator.generate('love', max_words=100, max_words_per_line=10, temperature=0.5)
# print(lyrics)
# # Output:
# # love me like you do
    