import torch
import torch.optim as optim
from data_loader import get_contrastive_loader
from model import ContrastiveModel
from train import train_one_epoch

def run_training():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 20
    BATCH_SIZE = 16
    LR = 1e-3

    model = ContrastiveModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loader = get_contrastive_loader("data/unlabeled", batch_size=BATCH_SIZE)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        loss = train_one_epoch(model, loader, optimizer, DEVICE)
        print(f"Average Loss: {loss:.4f}")
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")

if __name__ == '__main__':
    run_training()