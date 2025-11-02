import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

class CrashDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        # Load your CSV: expected columns ["vidname", "crash"] as built earlier
        self.labels_df = pd.read_csv(csv_file).head(1000)  # remove head(100) if you want full data
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        vid_plus_frame = self.labels_df.iloc[idx, 0]  # e.g., "000001-0"
        img_name = os.path.join(self.root_dir, vid_plus_frame + ".jpg")
        image = Image.open(img_name).convert('RGB')
        label = int(self.labels_df.iloc[idx, 1])  # 0/1
        
        if self.transform:
            image = self.transform(image)
        return image, label

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Full dataset
full_dataset = CrashDataset(csv_file="dataset.csv", root_dir="extracted_frames/", transform=transform)

# Stratified train/test split on indices using labels from the dataset
y = full_dataset.labels_df.iloc[:, 1].astype(int).values  # labels from CSV
indices = list(range(len(full_dataset)))
train_idx, test_idx = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=y
)

train_dataset = Subset(full_dataset, train_idx)
test_dataset  = Subset(full_dataset, test_idx)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# Model: use weights API (pretrained=True is deprecated)
model = resnet18(weights=ResNet18_Weights.DEFAULT)  # ImageNet weights
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # 2 classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 3

# Training
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss:.4f}")

# Path where the model will be saved
MODEL_PATH = 'model_weights.pth'

# 1. Save the model's learned state dictionary (weights)
# 2. Use map_location='cpu' to ensure cross-platform compatibility
#    (this is essential for loading it on your personal PC later)
torch.save(
    model.state_dict(), 
    MODEL_PATH, 
    _use_new_zipfile_serialization=False
)
print(f"\nModel weights saved successfully to {MODEL_PATH}")

# Evaluation on test set
model.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():  # disable autograd for speed/memory
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

avg_test_loss = test_loss / len(test_dataset)
test_acc = 100.0 * correct / total
print(f"Test Loss: {avg_test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")

# Evaluation on test set
model.eval()
test_loss = 0.0
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, 1)
        
        # Collect all labels and predictions for metric calculation
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# Calculate overall metrics
total = len(all_labels)
correct = sum(1 for i, j in zip(all_labels, all_preds) if i == j)
avg_test_loss = test_loss / total
test_acc = 100.0 * correct / total

print(f"\n--- Model Performance Report ---")
print(f"Test Loss: {avg_test_loss:.4f} | Overall Test Accuracy: {test_acc:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
print("\nClassification Report:")
# target_names should correspond to your labels (0: No Crash, 1: Crash)
print(classification_report(all_labels, all_preds, target_names=["No Crash", "Crash"]))