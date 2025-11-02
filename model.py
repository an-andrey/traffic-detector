import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

class CrashDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels_df = pd.read_csv(csv_file).head(100)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels_df.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = self.labels_df.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Imagenet stats
                        std=[0.229, 0.224, 0.225]),
])

train_dataset = CrashDataset(csv_file="dataset.csv", root_dir="data/extracted_frames", transform=None)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)  # Increase workers as per CPU

from torchvision import models
import torch.nn as nn

model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # binary classification: crash/no crash

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 3  # adjust based on time and validation performance

model.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss:.4f}")


