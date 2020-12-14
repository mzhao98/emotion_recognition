"""

Automatically generated by Colaboratory.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import os

# Download the data from Kaggle and unzip it
os.environ['KAGGLE_USERNAME'] = None # [REDACTED] username from the json file
os.environ['KAGGLE_KEY'] = None # [REDACTED] key from the json file
!kaggle datasets download -d msambare/fer2013
!mkdir emotion_data
!unzip -q fer2013.zip -d emotion_data/
!mkdir models

# GPU check
print(torch.cuda.device_count())
gpu = torch.device("cuda:0")

# Download models from Drive
from google.colab import drive
drive.mount('/content/drive')
!cp drive/MyDrive/models/2020-12-06_ensemble/*.pt models/
!cp drive/MyDrive/models/2020-12-06_ensemble/*.pkl models/

# Define some of the models
batch_size = 128

data_path = "emotion_data/"
classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
num_train = 28709
num_test = 7178
num_classes = len(classes)

# Transformations to be applied to the dataset
test_transforms = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])
test_ds = datasets.ImageFolder(data_path+"test", transform=test_transforms)
test_dataloader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# 9 models with this structure
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

# 1 model with this structure (this is the wide one)
class Net_v2(nn.Module):
    def __init__(self):
        super(Net_v2, self).__init__()
        # 1x48x48 input
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d((2,2))
        self.fc1 = nn.Linear(256*6*6, 4000)
        self.bn7 = nn.BatchNorm1d(4000)
        self.drop1 = nn.Dropout(0.65)
        self.fc2 = nn.Linear(4000, 1000)
        self.bn8 = nn.BatchNorm1d(1000)
        self.drop2 = nn.Dropout(0.65)
        self.fc3 = nn.Linear(1000, 200)
        self.bn9 = nn.BatchNorm1d(200)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(200, num_classes)
    
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
        x = F.relu(self.bn7(self.fc1(x))) # 800
        x = self.drop1(x)
        x = F.relu(self.bn8(self.fc2(x))) # 200
        x = self.drop2(x)
        x = F.relu(self.bn9(self.fc3(x))) # 200
        x = self.drop3(x)
        x = self.fc4(x) # 7

        return x

# Load up all of the models
models = []
models.append(torch.load("models/model1_normal.pt").to(gpu))
models.append(torch.load("models/model2_dark.pt").to(gpu))
models.append(torch.load("models/model3_bright.pt").to(gpu))
models.append(torch.load("models/model4_low_contrast.pt").to(gpu))
models.append(torch.load("models/model5_high_contrast.pt").to(gpu))
models.append(torch.load("models/model6_high_noise.pt").to(gpu))
models.append(torch.load("models/model7_blur.pt").to(gpu))
models.append(torch.load("models/model8_rot_left.pt").to(gpu))
models.append(torch.load("models/model9_rot_right.pt").to(gpu))

# Put all models in eval mode
for model in models:
  model.eval()

correct = 0
total = 0
y_true = torch.tensor([]).to(gpu) # Accumulated y_true and y_pred for confusion matrix
y_pred = torch.tensor([]).to(gpu)
# Iterate over test data
for i, data in enumerate(test_dataloader):
  inputs, labels = data[0].to(gpu), data[1].to(gpu)
  y_true = torch.cat((y_true, labels))
  outputs = 0
  for i, model in enumerate(models):  # Run the current batch through all models
    outputs += F.softmax(model(inputs), dim=1)
  y_hat = torch.argmax(outputs, 1)
  y_pred = torch.cat((y_pred, y_hat))
  correct += torch.sum(y_hat == labels)
  total += labels.size()[0]
print("Test Accuracy = %.5f" % (correct/total))

# Confusion matrix
import numpy as np
C = confusion_matrix(y_true.cpu().numpy(), y_pred.cpu().numpy()).astype(np.double)
for i in range(7):
  C[i,:] = C[i,:] / np.sum(C[i,:])
matplotlib.rcParams['figure.dpi']= 200
fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(C, cmap='Blues')
for (j,i),label in np.ndenumerate(C):
  ax.text(i,j,"%.2f" % label,ha='center',va='center', size='small')
plt.xticks(list(range(7)), classes, rotation=90)
plt.yticks(list(range(7)), classes)
fig.colorbar(im, ax=ax)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()