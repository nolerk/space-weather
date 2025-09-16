import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import logging

from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from src.datasets.data_preparation import train_loader
from src.datasets.time_series import TimeSeriesDataset
from src.utils.utils import PROJ_PATH

logger = logging.getLogger(__name__)

class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            out = self.fc(h_n[-1, :, :])
            return torch.sigmoid(out)

def train_and_save():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')

    input_size = 24
    hidden_size = 64
    num_layers = 2
    output_size = 1

    lstm_model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)

    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

    num_epochs = 40
    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        lstm_model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            data, labels = batch['x'].to(device), batch['y'].to(device)
            optimizer.zero_grad()
            output = lstm_model(data)
            loss = criterion(output, labels.view(-1, 1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_samples += len(labels) 
            predicted_labels = (output >= 0.5).float()
            correct_predictions += (predicted_labels == labels.view(-1, 1)).sum().item()

        average_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples
        train_losses.append(average_loss)
        train_accuracies.append(accuracy)

        logger.info(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}')
    
    model_path = PROJ_PATH / 'out' / 'model_checkpoints' / 'lstm.pth'
    torch.save(lstm_model.state_dict(), model_path)
    logger.info(f"Saved LSTM model to {model_path}")
