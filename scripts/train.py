import torch
import time
from pprint import pprint
import pickle
import argparse
import sys
sys.path.append('../')
from models.lstmmidi import LSTMMidiModel, LyricsMIDIDataset
from models.lstm import LSTMModel, LyricsDataset
from utils.train_utils import memory_stats, clear_memory, load_pretrained, train_model


"""
This script trains the LSTM model on the given dataset, using preprocessed concatenated embeddings.
"""


def main(config):
    pprint(config)
    print("Starting training...", flush=True)


    # clear cached memory if config['device'] is cuda
    memory_stats()
    clear_memory(config['device'])

    # load dataset
    print('Loading dataset...')
    if config['model'] == 'lstmmidi':     
        config["experiment_name"] = f"lstm_int_{config['hidden_dim']}_{config['num_layers']}_{config['learning_rate']}"
        preloaded_inputs = torch.load('data/dataset_saved_inputs.pt')
        preloaded_targets = torch.load('data/dataset_saved_targets.pt')
        preloaded_targets = torch.tensor(preloaded_targets, dtype=torch.long)  # Ensure targets are long
        dataset = LyricsMIDIDataset(preloaded_inputs=preloaded_inputs, preloaded_targets=preloaded_targets)
        
    
    elif config['model'] == 'lstm':
        with open('data/lyrics_dict.pkl', 'rb') as f:
            lyrics_dict = pickle.load(f)

        dataset = LyricsDataset(lyrics_dict)

    else:
        print(f"Invalid model type: {config['model']}")
        sys.exit(1)
        
    print("Dataset loaded successfully!")
    print(f"Number of training samples: {len(dataset)}")

    # initialize model
    input_dim = dataset.inputs.shape[-1]
    print("Input Dimension:", input_dim)
    print('Initializing LSTM model...')
    if config['model'] == 'lstmmidi':
        lstm_model = LSTMMidiModel(input_dim=input_dim, hidden_dim=config['hidden_dim'], vocab_size=config['vocab_size'], num_layers=config['num_layers']) 
    elif config['model'] == 'lstm':
        embedding_weights = torch.load('models/word2vec_weights.pt')
        vocab_size, embedding_dim = embedding_weights.shape
        lstm_model = LSTMModel(embedding_weights=embedding_weights, hidden_dim=256, vocab_size=vocab_size, num_layers=1)

    else:
        print(f"Invalid model type: {config['model']}")
        sys.exit(1)
    
    lstm_model = load_pretrained(lstm_model, f'models/{config["experiment_name"]}_weights.pth')

    # train
    print("Training...", flush=True)
    start = time.time()
    train_model(lstm_model, dataset, config)
    print(f"Training complete! Time taken: {time.time() - start:.2f} seconds")

    print("Done!")


# Configuration parameters
config = {
    'model': 'lstm',  # 'lstm' or 'lstmmidi'
    "batch_size": 64,
    "learning_rate": 0.001,
    "hidden_dim": 2,
    "num_layers": 2,
    "embedding_dim": 300,  # Assuming we know the embedding dimension
    "vocab_size": 3000000,  
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "shuffle": False,
    "epochs": 100,
    'verbosity': False
}

if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser(description="Training script for the Lyrics Generation model.")
    parser.add_argument("--model", default="lstm", type=str, help="The model to train: 'lstm' or 'lstmmidi'.")
    parser.add_argument("--batch_size", type=int, default=64, help="The batch size for training.")
    parser.add_argument("--device", type=str, default='cpu', help="The device to use for training: 'cpu' or 'cuda'.")
    parser.add_argument("--learningrate", type=float, default=0.001, help="The learning rate for training.")
    parser.add_argument("--epochs", type=int, default=10, help="The number of epochs to train for.")
    parser.add_argument("--hidden_dim", type=int, default=128, help="The number of features in the hidden state `h` of the LSTM.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of recurrent layers.")
    parser.add_argument("--verbosity", type=bool, default=False, help="Verbosity of the training process.")
    parser.add_argument("--save", type=bool, default=True, help="Save the model weights after training.")
    args = parser.parse_args()

    # update config
    config['model'] = args.model
    config['batch_size'] = args.batch_size
    config['device'] = torch.device(args.device)
    config['learning_rate'] = args.learningrate
    config['epochs'] = args.epochs
    config['hidden_dim'] = args.hidden_dim
    config['num_layers'] = args.num_layers
    config['verbosity'] = args.verbosity
    config['save'] = args.save

    # run
    main(config)
    

