import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, device, temperature=0.5):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc="Training")
    for view1, view2 in pbar:
        view1, view2 = view1.to(device), view2.to(device)
        z1 = model(view1)
        z2 = model(view2)
        
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        z = torch.cat([z1, z1], dim=0)
        sim = torch.mm(z, z.t()) / temperature
        
        n = z1.size(0)
        labels = torch.cat([torch.arange(n, 2*n), torch.arange(0, n)].to(device))
        
        loss = F.cross_entropy(sim, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"Loss" : loss.item()})
        
        wandb.log({"batch_loss" : loss.item()})
        
    return total_loss / len(loader)

wandb.init(project="EfficientXRAY", config={"lr" : 1e-3, "batch_size" : 32})