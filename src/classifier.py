import torch
import torch.nn as nn
from model import ContrastiveModel

class XRayClassifier(nn.Module):
    def __init__(self, checkpoint_path, num_classes=2):
        super(XRayClassifier, self).__init__()

        base_model = ContrastiveModel()
        base_model.load_state_dict(torch.load(checkpoint_path))
        self.encoder = base_model.encoder

        for param in base_model.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(2048, num_classes)

    
    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)