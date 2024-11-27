import torch
from torch import nn
from torch.nn import functional as F

class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        ##############################    
        # Showed error super(CustomCNN, self).__init__()
        # Define the convolutional layers
        # Convolutional layers and Batch Normalization
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(4 * 4 * 128, 512)
        self.bn7 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 10) 
        
        # MaxPooling и Dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout2d = nn.Dropout2d(p=0.2)
        self.dropout = nn.Dropout(p=0.3)
        ##############################



    def forward(self, x):
        # also I think we would do better if we added skip_connection like x = self.block(x) + x

        # first layer
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))

        # second layer
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))

        # third layer
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = F.relu(self.bn6(self.conv6(x)))

        # Dropout
        x = self.dropout2d(x)

        x = torch.flatten(x, 1)

        x = F.relu(self.bn7(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)  # out layer

        return x