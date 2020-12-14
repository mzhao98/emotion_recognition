"""model7_blur.ipynb

Automatically generated by Colaboratory.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
import os

# Import dataset directly from kaggle
os.environ['KAGGLE_USERNAME'] = None # [REDACTED] username from the json file
os.environ['KAGGLE_KEY'] = None # [REDACTED] key from the json file
!kaggle datasets download -d msambare/fer2013
!mkdir emotion_data
!unzip -q fer2013.zip -d emotion_data/
!mkdir models

# GPU check
print(torch.cuda.device_count())
gpu = torch.device("cuda:0")

# Run this if you want to bring a model from Drive onto the VM
from google.colab import drive
drive.mount('/content/drive')
!cp drive/MyDrive/model.pt models/

LOAD = True
batch_size = 256
learning_rate = 0.00001

data_path = "emotion_data/"
classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
num_train = 28709
num_test = 7178
num_classes = len(classes)

# Load training data
class AddNoise(object):
    def __init__(self, std):
        self.std = std
    def __call__(self, input):
        return input + torch.randn(input.shape)*self.std
class AdjustBrightness(object):
    def __init__(self, factor):
        self.factor = factor
    def __call__(self, input):
        return transforms.functional.adjust_brightness(input, self.factor)
class AdjustContrast(object):
    def __init__(self, factor):
        self.factor = factor
    def __call__(self, input):
        return transforms.functional.adjust_contrast(input, self.factor)
    
# Transformations to be applied to the dataset
train_transforms = transforms.Compose([
    transforms.Grayscale(),
    transforms.RandomHorizontalFlip(),
    # AdjustBrightness(1.4),  # 1.4 for bright, 0.6 for dark
    # AdjustContrast(1.6),  # 1.6 for high contrast, 0.5 for low contrast
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    # AddNoise(0.015),  # 0.015 for high noise
    transforms.Normalize([0.5],[0.5])
])
test_transforms = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

train_ds = datasets.ImageFolder(data_path+"train", transform=train_transforms)
train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_ds = datasets.ImageFolder(data_path+"test", transform=test_transforms)
test_dataloader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Define network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1x48x48 input
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 24, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 32, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 48, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(48)
        self.conv6 = nn.Conv2d(48, 64, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d((2,2))
        self.fc1 = nn.Linear(64*6*6, 800)
        self.drop1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(800, 200)
        self.drop2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(200, num_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))) # 8x48x48
        x = F.relu(self.bn2(self.conv2(x))) # 16x48x48
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x))) # 24x24x24
        x = F.relu(self.bn4(self.conv4(x))) # 32x24x24
        x = self.pool(x)
        x = F.relu(self.bn5(self.conv5(x))) # 48x12x12
        x = F.relu(self.bn6(self.conv6(x))) # 64x12x12
        x = self.pool(x)
        x = torch.flatten(x,1)  # 2304
        x = F.relu(self.fc1(x)) # 800
        x = self.drop1(x)
        x = F.relu(self.fc2(x)) # 200
        x = self.drop2(x)
        x = self.fc3(x) # 7

        return x

if LOAD:
    net = torch.load("models/model.pt")
else:
    net = Net()

net = net.to(gpu)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
opt = optim.Adam(net.parameters(), lr=learning_rate)

epochs = 12
report_freq = (num_train/batch_size)//4


# Train the network
for epoch in range(epochs):
    net.train()
    loss_avg = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(train_dataloader):
        inputs, labels = data[0].to(gpu), data[1].to(gpu)
        opt.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        opt.step()

        loss_avg += loss
        y_hat = torch.argmax(outputs, 1)
        correct += torch.sum(y_hat == labels)
        total += labels.size()[0]
        if i % report_freq == report_freq-1:
            print("[%d, %4d] loss = %.5f, accuracy = %.5f" % (epoch+1, i+1, loss_avg.item() / report_freq, correct/total))
            loss_avg = 0.0
            correct = 0
            total = 0
    
    net.eval()
    correct = 0
    total = 0
    for i, data in enumerate(test_dataloader):
        inputs, labels = data[0].to(gpu), data[1].to(gpu)
        outputs = net(inputs)
        y_hat = torch.argmax(outputs, 1)
        correct += torch.sum(y_hat == labels)
        total += labels.size()[0]
    print("Test Accuracy = %.5f" % (correct/total))
    torch.save(net, "models/model.pt")

# Save model
print("Saving...")
torch.save(net, "models/model.pt")

print("Done")

# Run this to save the model to Drive
from google.colab import drive
drive.mount('/content/drive')
!cp models/model.pt drive/MyDrive/models/2020-12-06_ensemble/model7_blur.pt
# !cp drive/MyDrive/Colab\ Notebooks/10701_CNN.ipynb drive/MyDrive/models/2020-12-06_ensemble/model2_dark.ipynb