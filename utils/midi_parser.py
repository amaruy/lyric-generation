import numpy as np
import pretty_midi
import os
import pickle
import sys
sys.path.append('../')

def extract_features(midi_path, max_length=2048):
    """
    Extracts musical features from a MIDI file.

    Parameters:
    midi_path (str): Path to the MIDI file.
    max_length (int): Maximum length for normalized pitch and velocity arrays.

    Returns:
    np.ndarray: A combined feature vector including normalized pitches, velocities, and mean chroma.
    """
    midi_data = pretty_midi.PrettyMIDI(midi_path)

    # Normalized Pitches
    pitches = [note.pitch for instrument in midi_data.instruments for note in instrument.notes]
    pitches_normalized = np.array(pitches) / 127.0  # MIDI pitch range
    pitches_feature = np.zeros(max_length)
    pitches_feature[:len(pitches_normalized)] = pitches_normalized[:max_length]

    # Normalized Velocities
    velocities = [note.velocity for instrument in midi_data.instruments for note in instrument.notes]
    velocities_normalized = np.array(velocities) / 127.0  # MIDI velocity range
    velocities_feature = np.zeros(max_length)
    velocities_feature[:len(velocities_normalized)] = velocities_normalized[:max_length]

    # Mean Chroma
    chroma = midi_data.get_chroma()
    chroma_mean = np.mean(chroma, axis=1)

    # Combine features into a single vector
    feature_vector = np.concatenate([pitches_feature, velocities_feature, chroma_mean])

    return feature_vector

def process_midi_files(directory, output_file):
    """
    Processes all MIDI files in a directory, extracting features and saving them.

    Parameters:
    directory (str): Directory containing MIDI files.
    output_file (str): File path to save the extracted feature vectors.
    """
    feature_vectors = {}

    for filename in os.listdir(directory):
        if filename.endswith(".mid"):
            midi_path = os.path.join(directory, filename)
            try:
                # Parsing artist and song names from the filename
                artist, song = os.path.basename(filename).replace('.mid', '').lower().split('_-_')
                song = f"{song} {artist}".replace('_', ' ')
                feature_vector = extract_features(midi_path)
                feature_vectors[song] = feature_vector
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Save the feature vectors to a file
    with open(output_file, 'wb') as f:
        pickle.dump(feature_vectors, f)

# Example usage
midi_directory = 'data/midi_files'
output_file = 'data/midi_feature_vectors.pkl'
process_midi_files(midi_directory, output_file)
