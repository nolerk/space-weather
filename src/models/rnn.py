import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import logging

from src.datasets.data_preparation import train_loader
from src.datasets.time_series import TimeSeriesDataset
from src.utils.utils import PROJ_PATH

logger = logging.getLogger(__name__)
  
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, hidden = self.rnn(x)
        out = self.fc(hidden[-1, :, :])  # Take the hidden state of the last time step
        out = self.sigmoid(out)
        return out

def train_and_save():
    input_size = 24
    hidden_size = 64
    output_size = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
    rnn_model = RNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size)  
    criterion = nn.BCELoss()
    optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)

    num_epochs = 40
    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        rnn_model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            data, labels = batch['x'], batch['y']
            optimizer.zero_grad()
            output = rnn_model(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_samples += labels.size(0)
            predicted_labels = (output >= 0.5).float()
            correct_predictions += (predicted_labels == labels).sum().item()

        average_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples

        train_losses.append(average_loss)
        train_accuracies.append(accuracy)

        logger.info(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}')
        
    model_path = PROJ_PATH / 'out' /'model_checkpoints' / 'rnn.pth'
    torch.save(rnn_model.state_dict(), model_path)
    logger.info(f"Saved RNN model to {model_path}")
