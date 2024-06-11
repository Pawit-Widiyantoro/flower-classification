import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.data import Data

class customCNN(nn.Module):
    def __init__(self):
        super(customCNN, self).__init__()
        # proses ekstraksi ciri (feature extraction)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.5)
        # proses pengenalan ciri
        self.fc1 = nn.Linear(128 * 12 * 12, 256)
        self.fc2 = nn.Linear(256, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout1(x)
        x = x.view(-1, 128 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def main():
    BATCH_SIZE = 4
    EPOCH = 10

    train_loader = DataLoader(Data(), batch_size=BATCH_SIZE, shuffle=True)

    model = customCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # proses training model dan evaluasi
    for epoch in range(EPOCH):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for batch, (src, trg) in enumerate(train_loader):
            pred = model(src)
            loss = criterion(pred, trg)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print("loss = ", loss.item())

            running_loss += loss.item()
            _, predicted = torch.max(pred, 1)
            correct_predictions += (predicted == trg).sum().item()
            total_predictions += trg.size(0)

        avg_loss = running_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        print(f'Epoch {epoch+1}/{EPOCH}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

if __name__=="__main__":
    main()
