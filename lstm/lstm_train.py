import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader


class RobotStateDataset(Dataset):
    def __init__(self, pt_file):
        data = torch.load(pt_file)
        self.features = torch.FloatTensor(data['states'])  # Shape: [N, seq_len, 6]
        self.targets = torch.FloatTensor(data['actions'])  # Shape: [N, output_size, 6]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class KinematicsLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=3, output_size=6, dropout=0.2):
        super(KinematicsLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,  # 6 features: x, y, orientation_x/y/z/w
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)  # 6 outputs: predicted next state
        )

    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output, hidden


class CustomLoss(nn.Module):
    def __init__(self, position_weight=1.0, orientation_weight=0.5):
        super(CustomLoss, self).__init__()
        self.position_weight = position_weight
        self.orientation_weight = orientation_weight

    def forward(self, outputs, targets):
        # Separate position and orientation components
        position_pred = outputs[:, :2]  # x, y
        position_true = targets[:, :2]

        orientation_pred = outputs[:, 2:]  # quaternion
        orientation_true = targets[:, 2:]

        # Calculate losses
        position_loss = torch.mean(torch.square(position_true - position_pred))
        orientation_loss = torch.mean(torch.square(orientation_true - orientation_pred))

        # Combined weighted loss
        total_loss = self.position_weight * position_loss + self.orientation_weight * orientation_loss
        return total_loss


def train_model(model, train_loader, val_loader, num_epochs=200, learning_rate=0.001, device='cuda'):
    model = model.to(device)
    criterion = CustomLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            outputs, _ = model(batch_features)
            loss = criterion(outputs, batch_targets.squeeze(1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)

                outputs, _ = model(batch_features)
                loss = criterion(outputs, batch_targets.squeeze(1))
                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}]')
            print(f'Train Loss: {train_loss:.6f}')
            print(f'Val Loss: {val_loss:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')

    history = {
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    torch.save(history, 'training_history.pt')

    return train_losses, val_losses


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load datasets
    train_dataset = RobotStateDataset('preprocessed_data/train.pt')
    val_dataset = RobotStateDataset('preprocessed_data/val.pt')
    test_dataset = RobotStateDataset('preprocessed_data/test.pt')

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize model
    model = KinematicsLSTM(
        input_size=6,  # x, y, orientation_x/y/z/w
        hidden_size=128,
        num_layers=3,
        output_size=6,  # predicted next state
        dropout=0.2
    )

    # Train model
    train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=150,
        learning_rate=0.001,
        device=device
    )

    return model, test_loader, device


if __name__ == "__main__":
    main()