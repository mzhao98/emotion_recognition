#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import time
import copy

import torch.nn as nn
import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch.nn as nn
from torchvision.models import resnet50


# ## Load in Dataset

# In[31]:


def load_dataset():
    train_dir = '../data/train/'
    test_dir = '../data/test/'
    categories = ['happy', 'sad', 'fear', 'surprise', 'neutral', 'angry', 'disgust']

    train_file_dictionary = {}
    train_imagefile_to_class_dictionary = {}
    for emotion in categories:
        train_file_dictionary[emotion] = []

    counter = 0
    for i in range(len(categories)):
        for subdir, dirs, files in os.walk(train_dir+categories[i]+'/'):
            for file in files:
                train_file_dictionary[categories[i]].append(train_dir+categories[i]+'/'+file)
                train_imagefile_to_class_dictionary[counter] = {}
                train_imagefile_to_class_dictionary[counter]['file'] = train_dir+categories[i]+'/'+file
                train_imagefile_to_class_dictionary[counter]['label'] = i
                counter += 1



    test_file_dictionary = {}
    test_imagefile_to_class_dictionary = {}
    for emotion in categories:
        test_file_dictionary[emotion] = []

    counter = 0
    for i in range(len(categories)):
        for subdir, dirs, files in os.walk(test_dir+categories[i]+'/'):
            for file in files:
                test_file_dictionary[categories[i]].append(test_dir+categories[i]+'/'+file)
                test_imagefile_to_class_dictionary[counter] = {}
                test_imagefile_to_class_dictionary[counter]['file'] = test_dir+categories[i]+'/'+file
                test_imagefile_to_class_dictionary[counter]['label'] = i
                counter += 1

    return train_imagefile_to_class_dictionary, test_imagefile_to_class_dictionary


# In[32]:


image_list = []
for filename in glob.glob(train_dir + '' + '/*.jpg'): #assuming gif
    im=Image.open(filename)
    image_list.append(im)


# ## Create Image Dataset

# In[33]:


class FacialEmotionDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, imagefile_to_class_dictionary, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            imagefile_to_class_dictionary (dictionary): Dictionary of image filenames to class for each emotion.
        """
#         self.root_dir = root_dir
        self.imagefile_to_class_dictionary = imagefile_to_class_dictionary
        self.transform = transforms.Compose(
                [
                    transforms.Resize((96, 96)),
                    transforms.ToTensor(),
#                     transforms.CenterCrop(10),
                 
                 transforms.Normalize((0.5), 
                                      (0.5))])


    def __len__(self):
        return len(self.imagefile_to_class_dictionary.keys())

    def __getitem__(self, idx):
#         print("idx", idx)
        path_to_image = self.imagefile_to_class_dictionary[idx]['file']
#         image = io.imread(path_to_image)
        image = Image.open(path_to_image)
        image = self.transform(image).float()
        label = int(self.imagefile_to_class_dictionary[idx]['label'])
        return image, label


# In[34]:


train_dataset = FacialEmotionDataset(train_dir, train_imagefile_to_class_dictionary)
test_dataset = FacialEmotionDataset(test_dir, test_imagefile_to_class_dictionary)


# ## Load in Resnet50

# In[65]:


from torchvision.models import resnet50
import math
model = resnet50(pretrained=True)

# gray_model = resnet50(pretrained=True)
# if init_weights is None:
#     # change the first layer to use 1x7x7-sized kernels instead of 3x7x7-sized kernels
w = torch.zeros((64, 1, 7, 7))
nn.init.kaiming_uniform_(w, a=math.sqrt(5))
# else:
#     w = init_weights
model.conv1.weight.data = w

for model_block in list(model.children())[:-3]:
    for param in model_block.parameters():
        param.requires_grad = False
        
# Reinitialize last layer to be 7 output features
num_classes = 7
conv_out_features = model.fc.in_features
model.fc = nn.Linear(conv_out_features, num_classes)


# In[66]:


model


# In[67]:


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for batch_idx, (inputs, labels) in enumerate(dataloaders):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


# In[68]:



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model.to(device)
feature_extract = False

params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)


# In[ ]:





# In[69]:


train_imagefile_to_class_dictionary, test_imagefile_to_class_dictionary = load_dataset()
train_dataset = FacialEmotionDataset(train_imagefile_to_class_dictionary)
test_dataset = FacialEmotionDataset(test_imagefile_to_class_dictionary)


print(len(train_dataset))
train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                          shuffle=True,
                                         )


# In[70]:


criterion = nn.CrossEntropyLoss()
num_epochs = 10

# Train and evaluate
model_ft, hist = train_model(model_ft, train_data_loader, criterion, optimizer_ft, num_epochs=num_epochs)


# In[ ]:




