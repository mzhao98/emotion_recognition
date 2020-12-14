from dependencies import *

class InitialNet(nn.Module):
  # Pytorch Network for Classifying Faces into Pos, Neutral, and Neg Emotions.
    def __init__(self):
        super(Initial_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3), stride=1, 
                               padding=1, dilation=1, groups=1, 
                               bias=True, padding_mode='reflect')
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=1, 
                               padding=1, dilation=1, groups=1, 
                               bias=True, padding_mode='reflect')
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, 
                               padding=1, dilation=1, groups=1, 
                               bias=True, padding_mode='reflect')
        
        self.conv4 = nn.Conv2d(64, 32, kernel_size=(3,3), stride=1, 
                               padding=1, dilation=1, groups=1, 
                               bias=True, padding_mode='reflect')
        
        
        self.drop1 = nn.Dropout(p=0.1)
        self.norm1 = nn.LayerNorm([48, 48])
        self.norm2 = nn.LayerNorm([24, 24])
        
        self.fc1 = nn.Linear(1152, 256)
        self.fc2 = nn.Linear(256, 96)
        self.fc3 = nn.Linear(96, 3)
        
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = self.norm1(x)
        
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = self.norm2(x)
        
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))

        x = F.max_pool2d(F.relu(self.conv4(x)), (2,2))

        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.drop1(x)
        
        x = self.fc3(x)
        x = F.relu(x)

        output = F.log_softmax(x, dim=1)
        return output



class PositiveNet(nn.Module):
  # Pytorch Network for Classifying Positive Emotions.
    def __init__(self):
        super(PositiveNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3), stride=1, 
                               padding=1, dilation=1, groups=1, 
                               bias=True, padding_mode='reflect')
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=1, 
                               padding=1, dilation=1, groups=1, 
                               bias=True, padding_mode='reflect')
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, 
                               padding=1, dilation=1, groups=1, 
                               bias=True, padding_mode='reflect')
        
        self.conv4 = nn.Conv2d(64, 32, kernel_size=(3,3), stride=1, 
                               padding=1, dilation=1, groups=1, 
                               bias=True, padding_mode='reflect')
        
        
        self.drop1 = nn.Dropout(p=0.1)
        self.norm1 = nn.LayerNorm([48, 48])
        self.norm2 = nn.LayerNorm([24, 24])
        
        self.fc1 = nn.Linear(1152, 256)
        self.fc2 = nn.Linear(256, 96)
        self.fc3 = nn.Linear(96, 2)
        
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = self.norm1(x)
        
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = self.norm2(x)
        
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))

        x = F.max_pool2d(F.relu(self.conv4(x)), (2,2))

        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.drop1(x)
        
        x = self.fc3(x)
        x = F.relu(x)

        output = F.log_softmax(x, dim=1)
        return output


class NegativeNet(nn.Module):
  # Pytorch Network for Classifying Negative Emotions.
    def __init__(self):
        super(NegativeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3), stride=1, 
                               padding=1, dilation=1, groups=1, 
                               bias=True, padding_mode='reflect')
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=1, 
                               padding=1, dilation=1, groups=1, 
                               bias=True, padding_mode='reflect')
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, 
                               padding=1, dilation=1, groups=1, 
                               bias=True, padding_mode='reflect')
        
        self.conv4 = nn.Conv2d(64, 32, kernel_size=(3,3), stride=1, 
                               padding=1, dilation=1, groups=1, 
                               bias=True, padding_mode='reflect')
        
        
        self.drop1 = nn.Dropout(p=0.1)
        self.norm1 = nn.LayerNorm([48, 48])
        self.norm2 = nn.LayerNorm([24, 24])
        
        self.fc1 = nn.Linear(1152, 256)
        self.fc2 = nn.Linear(256, 96)
        self.fc3 = nn.Linear(96, 4)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = self.norm1(x)        
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = self.norm2(x)        
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))

        x = F.max_pool2d(F.relu(self.conv4(x)), (2,2))

        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.drop1(x)
        
        x = self.fc3(x)
        x = F.relu(x)

        output = F.log_softmax(x, dim=1)
        return output
