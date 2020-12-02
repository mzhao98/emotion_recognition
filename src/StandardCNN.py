#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
# import matplotlib.pyplot as plt
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

import torch.nn as nn
import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

os.environ['KMP_DUPLICATE_LIB_OK']='True'
# import scipy.io as sio
# from os import listdir
# from os.path import isfile, join


# ## Load in Dataset

# In[2]:


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


# ## Create Image Dataset

# In[3]:


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


# ## Create CNN

# In[4]:


class FaceNet(nn.Module):
    def __init__(self):
        super(FaceNet, self).__init__()
        # torch.Size([256, 1, 64, 64])
        # 3 input image channel (RGB), #6 output channels, 4x4 kernel 
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3), stride=1, 
                               padding=1, dilation=1, groups=1, 
                               bias=True, padding_mode='reflect')
        
        self.conv2 = nn.Conv2d(32, 96, kernel_size=(3,3), stride=1, 
                               padding=1, dilation=1, groups=1, 
                               bias=True, padding_mode='reflect')
        
        self.conv3 = nn.Conv2d(96, 256, kernel_size=(3,3), stride=1, 
                               padding=1, dilation=1, groups=1, 
                               bias=True, padding_mode='reflect')
        
        
        self.drop1 = nn.Dropout(p=0.1)
        self.norm1 = nn.LayerNorm([48, 48])
        self.norm2 = nn.LayerNorm([24, 24])
        
        self.fc1 = nn.Linear(36864, 4096)
        self.fc2 = nn.Linear(4096, 1028)
        self.fc3 = nn.Linear(1028, 7)
        
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = self.norm1(x)
#         print(x.shape)
        
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = self.norm2(x)
#         print(x.shape)
        
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))
#         print(x.shape)

        x = torch.flatten(x, 1)
#         print(x.shape)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.drop1(x)
        
        x = self.fc3(x)
        x = F.relu(x)
#         x = self.drop1(x)
        
#         output = x
        output = F.log_softmax(x, dim=1)
        return output
        


# In[5]:


train_imagefile_to_class_dictionary, test_imagefile_to_class_dictionary = load_dataset()
train_dataset = FacialEmotionDataset(train_imagefile_to_class_dictionary)
test_dataset = FacialEmotionDataset(test_imagefile_to_class_dictionary)


# In[6]:


len(train_dataset)


# In[7]:


def train_FaceNet():
    train_imagefile_to_class_dictionary, test_imagefile_to_class_dictionary = load_dataset()
    train_dataset = FacialEmotionDataset(train_imagefile_to_class_dictionary)
    test_dataset = FacialEmotionDataset(test_imagefile_to_class_dictionary)


    print(len(train_dataset))
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                              shuffle=True,
                                             )
    
    # Parameters
    max_epochs = 10
    lr = 0.01
    momentum = 0.9

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print("cuda device = ", device)


    face_net = FaceNet().double()
    # Try different optimzers here [Adam, SGD, RMSprop]
    optimizer = optim.SGD(face_net.parameters(), lr=lr, momentum=momentum)


    training_losses = []

    # Generators
    training_set = train_dataset
    training_generator = train_data_loader

    loss_fn = torch.nn.NLLLoss()
    # loss_fn = nn.CrossEntropyLoss()

    face_net.train()

    # Loop over epochs
    print("Beginning Training..................")
    for epoch in range(max_epochs):
        # print("epoch: ", epoch)
        # Training
        total_epoch_loss = 0
        for batch_idx, (batch_data, batch_labels) in enumerate(train_data_loader):
            print('batch_idx = ', batch_idx)
            batch_data = batch_data.double()
            batch_labels = batch_labels

            predicted_output = face_net(batch_data)

            predicted_output = predicted_output.double()                                
            target_output = batch_labels

            # print(predicted_output)
            # print()
            # print(target_output)



            loss = loss_fn(predicted_output, target_output)
    #         loss = F.nll_loss(predicted_output, target_output)   # Compute loss

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()  

            total_epoch_loss += loss.item()

            if batch_idx % 25 == 0:
                print('Train Epoch: {} \tLoss: {:.6f}'.format(
                    epoch, total_epoch_loss))

        if epoch % 100 == 0:
            with open('../saved_models/face_network_1.pkl', 'wb') as f:
                torch.save(face_net.state_dict(), f)

        training_losses.append(total_epoch_loss)

    with open('../saved_models/face_network_1_final.pkl', 'wb') as f:
        torch.save(face_net.state_dict(), f)

    with open('../saved_models/face_network_1_losses.npy', 'wb') as f:
        np.save(f, np.array(training_losses))


# In[ ]:


train_FaceNet()


# In[ ]:





# In[ ]:




