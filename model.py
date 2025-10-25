import torch
import torch.nn as nn
import torchvision.models as models

class RetinaNet(nn.Module):
    def __init__(self, num_classes=5):
        super(RetinaNet, self).__init__()
        self.model = models.efficientnet_b0(pretrained=True)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)
