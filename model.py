import torch
import torch.nn as nn
import torchvision
from torchsummary import summary
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(
            nn.Conv2d(1,6,kernel_size=5,padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(2,stride=2),
            nn.Conv2d(6,16,kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(2,stride=2),
            nn.Flatten(),
            nn.Linear(400,120),
            nn.Sigmoid(),
            nn.Linear(120,84),
            nn.Sigmoid(),
            nn.Linear(84,10)
        )
    def forward(self,input):
        output=self.model(input)
        return output


