import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loader(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    
    return train_loader, val_loader

if __name__ == "__main__":
    
    DATA_PATH = "data/train"
    try:
        train_loader, val_loader = get_data_loader(DATA_PATH)
        images, labels = next(iter(train_loader))
        print(f"Data loaded successfully - Batch size = {images.shape}")
        print(f"Number of classes : {len(train_loader.dataset.dataset.classes)}")
    except Exception as e:
        print(f"Error : {e}.")
    
    