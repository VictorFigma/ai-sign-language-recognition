import os
import cv2
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load labels
labels_df = pd.read_csv('data/labels_path_train.csv')

# Extract frames from videos
def extract_frames(video_path, num_frames=16, frame_size=(64, 64)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // num_frames)
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, frame_size)
        frames.append(frame)
    cap.release()
    while len(frames) < num_frames:
        frames.append(np.zeros((frame_size[0], frame_size[1], 3), dtype=np.uint8))
    return np.array(frames)

# Dataset
class SignLanguageDataset(Dataset):
    def __init__(self, df, num_frames=16, frame_size=(64, 64)):
        self.df = df
        self.num_frames = num_frames
        self.frame_size = frame_size
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        video_path = self.df.iloc[idx]['path']
        label = self.df.iloc[idx]['label']
        frames = extract_frames(video_path, self.num_frames, self.frame_size)
        frames = frames.transpose((3, 0, 1, 2))
        return torch.tensor(frames, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
    
# Data splitting
train_df, val_df = train_test_split(labels_df, test_size=0.2, random_state=42)

# Data loaders
train_dataset = SignLanguageDataset(train_df)
val_dataset = SignLanguageDataset(val_df)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, pin_memory=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, pin_memory=True, num_workers=0)

# Model architecture
class SignLanguageModel(nn.Module):
    def __init__(self, num_classes=10):
        super(SignLanguageModel, self).__init__()
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.fc1 = nn.Linear(64 * 4 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training
model = SignLanguageModel(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 7
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Validation Loss: {val_loss/len(val_loader)}, Accuracy: {100 * correct/total}%")

# Predict
def generate_predictions(model, test_videos_path):
    model.eval()
    predictions = {}
    with torch.no_grad():
        for video_file in os.listdir(test_videos_path):
            video_path = os.path.join(test_videos_path, video_file)
            frames = extract_frames(video_path)
            frames = frames.transpose((3, 0, 1, 2))
            frames = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(device)
            outputs = model(frames)
            _, predicted = torch.max(outputs, 1)
            predictions[video_file.split('.')[0]] = int(predicted.item())
    return predictions

test_videos_path = 'test/'
predictions = generate_predictions(model, test_videos_path)

# Save predictions
with open('predictions/predictions.json', 'w') as f:
    json.dump({"target": predictions}, f, indent=4)