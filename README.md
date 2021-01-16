CMU 10-701: Facial Expression Recognition on https://www.kaggle.com/msambare/fer2013

# Facial Expression Recognition

Facial expression recognition (FER) has been extensively studied given its importance in non-verbal communication. It has a wide range of applications such as pain detection in the medical field, drowsiness detection in driver safety, or facial action in the animation industry, among others. FER is a hard problem to solve given that in real world applications people have a variety of colors, skin textures, races, poses, etc. To tackle this problem, we created an algorithm that uses the FER2013 dataset to correctly label facial expressions from seven different emotions: happy, sad, angry, disgust, neutral, surprise, and fear. We trained nine identically-structured convolutional neural networks 
(CNNs) with modified versions of the training dataset by adding different image transformations to each one, we then trained a wide standalone CNN, and a standalone 4-CNN using augmented data by adding to the training set cropped versions, horizontal flips, oversampling to twice the size and normalization to each image. The models were combined in a final ensemble as a weighted sum of the individual network outputs. Our final test accuracy was 67.7\%. This result outperformed a human performance baseline.

The link to the associated report is https://github.com/mzhao98/emotion_recognition/blob/main/written.pdf


## Requirements:
- To run locally:
    - Python 3.7+ installed
    - Pytorch, Numpy, Matplotlib

## Multistep CNN
1. Run python Multistep_CNN.py

## Standalone 4-Conv Layer CNN Implementation
1. Run python standalone_4conv_cnn.py

