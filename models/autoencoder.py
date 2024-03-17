import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from torch.utils.tensorboard import SummaryWriter

class Autoencoder(nn.Module):
    def __init__(self, input_size, embedding_dim=1024):
        """
        Initializes the Autoencoder model with a specified input size and embedding dimension.

        Parameters:
        - input_size (int): The size of the input feature vector.
        - embedding_dim (int): The size of the embedding vector.
        """
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 2048),
            nn.ReLU(True),
            nn.Linear(2048, embedding_dim),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 4096),
            nn.ReLU(True),
            nn.Linear(4096, input_size),
            nn.Sigmoid(),  # Sigmoid activation to ensure output values are between 0 and 1
        )

    def forward(self, x):
        """
        Forward pass of the Autoencoder. Encodes and then decodes the input.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Reconstructed input tensor.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        """
        Encodes the input into a lower-dimensional embedding.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Encoded (embedded) tensor.
        """
        return self.encoder(x)

class VectorDataset(Dataset):
    def __init__(self, vector_dict):
        """
        Initializes the dataset with vectors extracted from MIDI files.

        Parameters:
        - vector_dict (dict): Dictionary containing feature vectors.
        """
        self.vectors = list(vector_dict.values())

    def __len__(self):
        """
        Returns the number of items in the dataset.
        """
        return len(self.vectors)

    def __getitem__(self, idx):
        """
        Retrieves an item by its index.

        Parameters:
        - idx (int): Index of the item.

        Returns:
        - torch.Tensor: Feature vector as a tensor.
        """
        vector = self.vectors[idx]
        return torch.tensor(vector, dtype=torch.float)

def train_autoencoder(autoencoder, feature_vectors, criterion, optimizer, num_epochs=100, device=torch.device("cpu")):
    """
    Trains the autoencoder model.

    Parameters:
    - autoencoder (Autoencoder): The autoencoder model.
    - feature vectors (Dict): Midi feature vectors for the songs {song:vector}.
    - criterion (torch.nn.modules.loss): Loss function.
    - optimizer (torch.optim.Optimizer): Optimizer.
    - num_epochs (int): Number of epochs to train.
    - device (torch.device): Device to train on.
    """
    dataset = VectorDataset(feature_vectors)
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=True)
    # write to tensorboard
    writer = SummaryWriter('runs/autoencoder_midi')
    autoencoder.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data in dataloader:
            inputs = data.to(device)
            optimizer.zero_grad()
            outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Log the average loss for the epoch
        epoch_loss = running_loss / len(dataloader)
        writer.add_scalar('training_loss', epoch_loss, epoch+1)
        if (epoch+1) % 1000 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
   
    return autoencoder

def load_pretrained_autoencoder(autoencoder, weight_path):
    """
    Loads pretrained weights into the autoencoder model.

    Parameters:
    - autoencoder (Autoencoder): The autoencoder model.
    - weight_path (str): Path to the pretrained weights file.

    Returns:
    - Autoencoder: The autoencoder model with pretrained weights.
    """
    try:
        autoencoder.load_state_dict(torch.load(weight_path))
        print(f"Loaded autoencoder weights from {weight_path}")
    except FileNotFoundError:
        print(f"No autoencoder weights found at {weight_path}")
    return autoencoder

def generate_embeddings(autoencoder, feature_vectors, device=torch.device("cpu")):
    """
    Generates embeddings for the given feature vectors using the trained autoencoder.

    Parameters:
    - autoencoder (Autoencoder): The trained autoencoder model.
    - feature_vectors (Dict): Midi feature vectors for the songs {song:vector}.
    - device (torch.device): Device to generate embeddings on.

    Returns:
    - Dict: Dictionary containing the song names as keys and their embeddings as values.
    """
    autoencoder.eval()
    embeddings = {}
    for song, vector in feature_vectors.items():
        input_vector = torch.tensor(vector, dtype=torch.float).to(device)
        encoded_vector = autoencoder.encode(input_vector).detach().cpu().numpy()
        embeddings[song] = encoded_vector
    return embeddings   



