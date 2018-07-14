import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.batch_norm = nn.BatchNorm2d(1)
        self.activation = F.elu
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(p=0.2)
        
        self.conv = [
            nn.Conv2d(1, 24, 5, padding=1),
            nn.Conv2d(24, 36, 5, padding=1),
            nn.Conv2d(36, 48, 5, padding=1),
            nn.Conv2d(48, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1)
        ]
        
        self.conv1 = self.conv[0]
        self.conv2 = self.conv[1]
        self.conv3 = self.conv[2]
        self.conv4 = self.conv[3]
        self.conv5 = self.conv[4]
        
        self.fc = [
            nn.Linear(64*5*5, 500),
            nn.Linear(500, 250),
            nn.Linear(250, 136)
        ]
        
        self.fc1 = self.fc[0]
        self.fc2 = self.fc[1]
        self.fc3 = self.fc[2]
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))

        x = self.batch_norm(x)
        
        num_convs = len(self.conv)
        
        for index in range(num_convs):
            x = self.activation(self.conv[index](x))
            if index < (num_convs - 1):
                x = self.dropout(self.pool(x))
        
        x = x.view(-1, 64*5*5)
        
        num_fcs = len(self.fc)
        for index in range(num_fcs):
            x = self.fc[index](x)
            if index < num_fcs - 1:
                x = self.activation(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
