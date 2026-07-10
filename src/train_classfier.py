import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from data_loader import LabeledDataset
from classifier import XRayClassifier
from torchvision.datasets import ImageFolder

def train_classifier():

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = XRayClassifier(checkpoint_path="checkpoint_epoch_20.pth", num_classes=2)
    model = model.to(DEVICE)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(root="data/labeled/train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)

    model.train()

    for epoch in range(10):
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} finished, Loss : {loss.item():.4f}")

    torch.save(model.state_dict(), "final_xray_classifier.pth")

if __name__ == "__main__":
    train_classifier()