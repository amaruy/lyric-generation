import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pickle as pkl
from time import time
import sys

import pretty_midi
from gensim.models import KeyedVectors
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation
import gensim.downloader as api

from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from itertools import product

start = time()

word2vec_path = r'models/GoogleNews-vectors-negative300.bin.gz'
word2vec_model = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

class LyricsMelodyDataset(Dataset):

    def __init__(self, df, word2vec_model, segment_size=4, sequence_length=50, df_size=1000):
        self.df = df[:df_size]
        self.word2vec_model = word2vec_model
        self.segment_size = segment_size
        self.sequence_length = sequence_length
        
        self.inputs = []
        self.melodies = []
        self.targets = []
        
        self._create_sequences_and_melodies()
        self._standardize_melodies()

    def preprocess_text(self, text):
        text = text.lower()
        text = remove_stopwords(text)
        text = strip_punctuation(text)
        return text

    def text_to_vector(self, text):
        words = self.preprocess_text(text).split()
        indices = [self.word2vec_model.key_to_index[word] for word in words if word in self.word2vec_model.key_to_index]
        vectors = [self.word2vec_model[word] for word in words if word in self.word2vec_model.key_to_index]
        return np.array(vectors), indices


    def _create_sequences_and_melodies(self):
        for index, row in tqdm(self.df.iterrows(), total=len(self.df)):
            lyrics_vectors, lyrics_indices = self.text_to_vector(row['lyrics'])
            total_segments = int(np.ceil(len(lyrics_vectors) / self.segment_size))
            melody_vectors = self.vectorize_midi_segments(row['midi'], total_segments)
            
            expanded_melody_vectors = np.repeat(melody_vectors, repeats=self.segment_size, axis=0)[:len(lyrics_vectors)]

            for i in range(0, len(lyrics_vectors) - self.sequence_length, self.segment_size):
                input_sequence = lyrics_vectors[i:i+self.sequence_length]
                melody_sequence = expanded_melody_vectors[i:i+self.sequence_length]
                
                target_sequence = lyrics_indices[i+1:i+self.sequence_length+1]

                self.inputs.append(input_sequence)
                self.melodies.append(melody_sequence)
                self.targets.append(target_sequence)
            
    def _standardize_melodies(self):
        all_melodies_flat = np.vstack(self.melodies)
        scaler = StandardScaler()
        all_melodies_flat_standardized = scaler.fit_transform(all_melodies_flat)
        
        num_samples = len(self.melodies)
        self.melodies = all_melodies_flat_standardized.reshape(num_samples, self.sequence_length, -1)

    def segment_midi(self, midi_data, n_segments):
        total_duration = midi_data.get_end_time()
        segment_duration = total_duration / n_segments
        segments = []

        for i in range(n_segments):
            start_time = i * segment_duration
            end_time = start_time + segment_duration
            segments.append((start_time, end_time))
        
        return segments

    def extract_features_for_segment(self, midi_data, start_time, end_time, fs=100, pedal_threshold=64):
        notes = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                if start_time <= note.start < end_time:
                    notes.append(note)
        
        # Calculate average pitch and velocity, and total duration of notes in the segment
        if notes:
            average_pitch = np.mean([note.pitch for note in notes])
            average_velocity = np.mean([note.velocity for note in notes])
            total_duration_notes = sum([note.end - note.start for note in notes if note.start < end_time and note.end > start_time])
        else:
            average_pitch = 0
            average_velocity = 0
            total_duration_notes = 0

        # Create one-hot vector for instrument presence
        instrument_vector = np.zeros(128)  # Assuming General MIDI
        for instrument in midi_data.instruments:
            if any(start_time <= note.start < end_time for note in instrument.notes):
                instrument_vector[instrument.program] = 1
        
        # Calculate chroma features for the segment
        chroma = midi_data.get_chroma(fs=fs, times=np.arange(start_time, end_time, 1./fs), pedal_threshold=pedal_threshold)
        average_chroma = np.mean(chroma, axis=1)  # Averaging chroma vectors over the segment

        # Concatenate all features into a single vector
        features = np.concatenate(([average_pitch, average_velocity, total_duration_notes], instrument_vector, average_chroma))
        
        return features

    def vectorize_midi_segments(self, midi_data, n_segments):
        segments = self.segment_midi(midi_data, n_segments)
        feature_vectors = []

        for start_time, end_time in segments:
            features = self.extract_features_for_segment(midi_data, start_time, end_time)
            feature_vectors.append(features)
    
        return np.array(feature_vectors)

    def __len__(self):
        return len(self.inputs)
     
    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx], dtype=torch.float),
            torch.tensor(self.melodies[idx], dtype=torch.float),
            torch.tensor(self.targets[idx], dtype=torch.long)  # Adjusted to torch.long
    )

class LyricsMelodyLSTM(nn.Module):
    def __init__(self, text_input_dim, melody_input_dim, hidden_dim, output_dim, num_layers=2, dropout=0, flag=False):
        super(LyricsMelodyLSTM, self).__init__()
        self.flag = flag

        if flag:
            self.text_lstm = nn.LSTM(text_input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
            self.melody_lstm = nn.LSTM(melody_input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_dim * 2, output_dim)
        else:
            self.lstm = nn.LSTM(text_input_dim + melody_input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text, melody, hidden=None):
        if self.flag:
            text_output, text_hidden = self.text_lstm(text)
            melody_output, melody_hidden = self.melody_lstm(melody)
            combined = torch.cat((text_output, melody_output), dim=2)
        else:
            combined_input = torch.cat((text, melody), dim=2)
            combined_output, combined_hidden = self.lstm(combined_input, hidden)
            combined = combined_output
        
        output = self.fc(combined)
        return output

def train_model(model, train_dataloader, val_dataloader, epochs=3, lr=2e-5):
    train_losses = []
    val_losses = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (text, melody, targets) in enumerate(train_dataloader):
            optimizer.zero_grad()
            text, melody, targets = text.to(device), melody.to(device), targets.to(device)
            outputs = model(text, melody)
            targets = targets.view(-1)
            outputs = outputs.view(-1, outputs.shape[-1])

            loss = criterion(outputs, targets.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_loss = evaluate_model(model, val_dataloader)
            print(f'Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}')
            val_losses.append(val_loss)
        model.train()

        train_losses.append(running_loss / len(train_dataloader))

    return train_losses, val_losses

def evaluate_model(model, dataloader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for i, (text, melody, targets) in enumerate(dataloader):
            text, melody, targets = text.to(device), melody.to(device), targets.to(device)
            outputs = model(text, melody)
            targets = targets.view(-1)
            outputs = outputs.view(-1, outputs.shape[-1])
            loss = criterion(outputs, targets.long())
            total_loss += loss.item()
    return total_loss / len(dataloader) 

results_path = 'results/results.csv'

search = int(sys.argv[1])
segments = [1,4,8]
sequence_lengths = [50,100,200]
segment, sequence_length = list(product(segments, sequence_lengths))[search]
print(f'Running search {search} with segment size {segment} and sequence length {sequence_length}')

epochs = 30
batch_size = 1
learning_rate = 0.001

data_set_name = f'seg{segment}_seq{sequence_length}.pkl'
train_dataset_path = r'data' + data_set_name
val_dataset_path = r'data' + data_set_name
test_dataset_path = r'data' + data_set_name

print('LOADING DATASETS!')

try:
    with open(train_dataset_path, 'rb') as f:
        train_dataset = pkl.load(f)
    with open(val_dataset_path, 'rb') as f:
        val_dataset = pkl.load(f)
    with open(test_dataset_path, 'rb') as f:
        test_dataset = pkl.load(f)
except:
    print('Creating new datasets')
    train_path = r'data/lyrics_train_set.csv'
    test_path = r'data/lyrics_test_set.csv'

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train['artist'] = train['artist'].str.strip()
    train['song'] = train['song'].str.strip()
    test['artist'] = test['artist'].str.strip()
    test['song'] = test['song'].str.strip()

    midi_path = r'data/midi_files'

    def load_midi_files(path):
        midi_files = {}
        tuples = []
        for filename in os.listdir(path):
            if filename.endswith(".mid"):
                file_path = os.path.join(path, filename)
                try:
                    names = filename.split('-')
                    artist = names[0].replace('_', ' ').strip().lower()
                    song = names[1].replace('_', ' ').strip().lower()
                    if song[-4:] == '.mid':
                        song = song[:-4]
                    midi = pretty_midi.PrettyMIDI(file_path)
                    midi_files[(artist, song)] = midi
                    tuples.append((artist, song))
                except:
                    print(f"Could not load {file_path} as a midi file. Skipping.")
        return midi_files, tuples
    
    midi_files, tuples = load_midi_files(midi_path)

    train['midi'] = train.apply(lambda row: midi_files.get((row['artist'].lower(), row['song'].lower())), axis=1)
    test['midi'] = test.apply(lambda row: midi_files.get((row['artist'].lower(), row['song'].lower())), axis=1)
    train = train.dropna(subset=['midi'])
    train, val = train_test_split(train, test_size=0.1, random_state=42)

    train_dataset = LyricsMelodyDataset(train, word2vec_model, segment_size=segment, sequence_length=sequence_length)
    val_dataset = LyricsMelodyDataset(val, word2vec_model, segment_size=segment, sequence_length=sequence_length)
    test_dataset = LyricsMelodyDataset(test, word2vec_model, segment_size=segment, sequence_length=sequence_length)


text_input_dim = 300
melody_input_dim = 143
hidden_dim = 256
output_dim = len(word2vec_model)
num_layers = 2
dropout = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LyricsMelodyLSTM(text_input_dim, melody_input_dim, hidden_dim, output_dim, num_layers, dropout, flag=False).to(device)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print('TRAINING MODEL!')
train_losses, val_losses = train_model(model, train_dataloader, val_dataloader, epochs=epochs, lr=learning_rate)
train_losses = '-'.join([str(round(x, 4)) for x in train_losses])
val_losses = '-'.join([str(round(x, 4)) for x in val_losses])

time_elapsed = time() - start

results_dict = {'search': search,'segment_size': [segments], 'sequence_length': [sequence_length], 'epochs': [epochs],
                'batch_size': [batch_size],'learning_rate': [learning_rate], 'num_layers': [num_layers], 'dropout': [dropout],
                'time_elapsed': [time_elapsed], 'train_losses': [train_losses], 'val_losses': [val_losses]}

curr_results_df = pd.DataFrame(results_dict)


# if results_path exists, add to it, else create it
if os.path.exists(results_path):
    results_df = pd.read_csv(results_path)
    results_df = pd.concat([results_df, curr_results_df], ignore_index=True)
    results_df.to_csv(results_path, index=False)

else:
    curr_results_df.to_csv(results_path, index=False)


