import torch
import torch.nn as nn
from torchvision import models

class ContrastiveModel(nn.Module):
    def __init__(self):
        super(ContrastiveModel, self).__init__()
        
        self.encoder = models.resnet50(pretrained=True)
        in_features = self.encoder.fc.in_features
        
        self.encoder.fc = nn.Indentity()
        
        self.projection_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        
    def forward(self, x):
        features = self.encoder(x)
        return self.projection_head(features)