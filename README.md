CMU 10-701: Facial Expression Recognition on https://www.kaggle.com/msambare/fer2013

# Image transformation ensemble
Files for this ensemble are contained in the "transform_ensemble" folder.
Files named "model[x]_[name].py" contain the training code for the nine component models
These files would train the networks and create .pt files containing the model
The "transform_ensemble_test.py" file contains the code to import and test all models
This file would open all of the selected .pt files containing the models and evaluate them on the test set as an ensemble
Note: all code was developed in Google Colab, and cannot be executed without access to personal Google Drive folders and Kaggle accounts

# Wide standalone CNN
Training code for this model is contained in the "wide_model.py"
Note: all code was developed in Google Colab, and cannot be executed without access to personal Google Drive folders and Kaggle accounts

# Weighted voting ensemble
Evaluation code for this model is contained in "weighted_ensemble_test.py"
This file assumes that the models have already been trained, and can be imported from .pt and .pkl files
Note: all code was developed in Google Colab, and cannot be executed without access to personal Google Drive folders and Kaggle accounts


# Multistep CNN and Standalone 4-Conv Layer CNN Implementation


## Requirements:
- To run locally:
    - Python 3.7+ installed
    - Pytorch, Numpy, Matplotlib

## Multistep CNN
1. Run python Multistep_CNN.py

## Standalone 4-Conv Layer CNN Implementation
1. Run python standalone_4conv_cnn.py

