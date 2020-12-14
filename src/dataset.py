from dependencies import *


def load_dataset_orig():
    """
    This function creates a dictionary of image filenames to class for each image in the FER2013 dataset.
    Outputs:
        train_imagefile_to_class_dictionary: (dictionary): Dictionary of training image filenames to class 
        for each emotion.
        test_imagefile_to_class_dictionary (dictionary): Dictionary of testing image filenames to class 
        for each emotion.
    """
    train_dir = 'emotion_recognition/data/train/'
    test_dir = 'emotion_recognition/data/test/'
    categories = ['happy', 'sad', 'fear', 'surprise', 'neutral', 'angry', 'disgust']
    positive = ['happy', 'surprise']
    neither = [ 'neutral']
    negative = ['sad', 'fear', 'angry', 'disgust']

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


def load_positive_dataset():
    """
    This function creates a dictionary of image filenames to class for each image in the FER2013 dataset 
    corresponding to only positive emotions: Happy and Surprised.
    Outputs:
        train_imagefile_to_class_dictionary: (dictionary): Dictionary of training image filenames to class 
        for each emotion.
        test_imagefile_to_class_dictionary (dictionary): Dictionary of testing image filenames to class 
        for each emotion.
    """
    train_dir = 'emotion_recognition/data/train/'
    test_dir = 'emotion_recognition/data/test/'
    categories = ['happy', 'sad', 'fear', 'surprise', 'neutral', 'angry', 'disgust']
    positive = ['happy', 'surprise']
    neither = [ 'neutral']
    negative = ['sad', 'fear', 'angry', 'disgust']

    train_file_dictionary = {}
    train_imagefile_to_class_dictionary = {}
    for emotion in categories:
        train_file_dictionary[emotion] = []

    counter = 0
    for i in range(len(positive)):
        for subdir, dirs, files in os.walk(train_dir+positive[i]+'/'):
            for file in files:
                train_file_dictionary[positive[i]].append(train_dir+positive[i]+'/'+file)
                train_imagefile_to_class_dictionary[counter] = {}
                train_imagefile_to_class_dictionary[counter]['file'] = train_dir+positive[i]+'/'+file
                train_imagefile_to_class_dictionary[counter]['label'] = i
                counter += 1



    test_file_dictionary = {}
    test_imagefile_to_class_dictionary = {}
    for emotion in categories:
        test_file_dictionary[emotion] = []

    counter = 0
    for i in range(len(positive)):
        for subdir, dirs, files in os.walk(test_dir+positive[i]+'/'):
            for file in files:
                test_file_dictionary[positive[i]].append(test_dir+positive[i]+'/'+file)
                test_imagefile_to_class_dictionary[counter] = {}
                test_imagefile_to_class_dictionary[counter]['file'] = test_dir+positive[i]+'/'+file
                test_imagefile_to_class_dictionary[counter]['label'] = i
                counter += 1

    return train_imagefile_to_class_dictionary, test_imagefile_to_class_dictionary



def load_negative_dataset():
    """
    This function creates a dictionary of image filenames to class for each image in the FER2013 dataset 
    corresponding to only negative emotions: Sad, Fear, Anger, Disgust
    Outputs:
        train_imagefile_to_class_dictionary: (dictionary): Dictionary of training image filenames to class 
        for each emotion.
        test_imagefile_to_class_dictionary (dictionary): Dictionary of testing image filenames to class 
        for each emotion.
    """
    train_dir = 'emotion_recognition/data/train/'
    test_dir = 'emotion_recognition/data/test/'
    categories = ['happy', 'sad', 'fear', 'surprise', 'neutral', 'angry', 'disgust']
    positive = ['happy', 'surprise']
    neither = [ 'neutral']
    negative = ['sad', 'fear', 'angry', 'disgust']

    train_file_dictionary = {}
    train_imagefile_to_class_dictionary = {}
    for emotion in categories:
        train_file_dictionary[emotion] = []

    counter = 0
    for i in range(len(negative)):
        for subdir, dirs, files in os.walk(train_dir+negative[i]+'/'):
            for file in files:
                train_file_dictionary[negative[i]].append(train_dir+negative[i]+'/'+file)
                train_imagefile_to_class_dictionary[counter] = {}
                train_imagefile_to_class_dictionary[counter]['file'] = train_dir+negative[i]+'/'+file
                train_imagefile_to_class_dictionary[counter]['label'] = i
                counter += 1



    test_file_dictionary = {}
    test_imagefile_to_class_dictionary = {}
    for emotion in categories:
        test_file_dictionary[emotion] = []

    counter = 0
    for i in range(len(negative)):
        for subdir, dirs, files in os.walk(test_dir+negative[i]+'/'):
            for file in files:
                test_file_dictionary[negative[i]].append(test_dir+negative[i]+'/'+file)
                test_imagefile_to_class_dictionary[counter] = {}
                test_imagefile_to_class_dictionary[counter]['file'] = test_dir+negative[i]+'/'+file
                test_imagefile_to_class_dictionary[counter]['label'] = i
                counter += 1

    return train_imagefile_to_class_dictionary, test_imagefile_to_class_dictionary




class FacialEmotionDataset(Dataset):
    def __init__(self, imagefile_to_class_dictionary, transform=None):
        """
        Create dataset of original images.
        Args:
            root_dir (string): Directory with all the images.
            imagefile_to_class_dictionary (dictionary): Dictionary of image filenames to class for each emotion.
        """
        self.imagefile_to_class_dictionary = imagefile_to_class_dictionary
        self.transform = transforms.Compose(
                [
                    transforms.Resize((96, 96)),
                    transforms.ToTensor(),
                 transforms.Normalize((0.5), 
                                      (0.5))])


    def __len__(self):
        return len(self.imagefile_to_class_dictionary.keys())

    def __getitem__(self, idx):
        path_to_image = self.imagefile_to_class_dictionary[idx]['file']
        image = Image.open(path_to_image)
        image = self.transform(image).float()
        label = int(self.imagefile_to_class_dictionary[idx]['label'])
        return image, label


class FacialEmotionDataset_Augmented(Dataset):
    def __init__(self, imagefile_to_class_dictionary, transform=None):
        """
        Create dataset of perturbed images (using horizontal flip and random crop) to augment 
        original dataset.
        Args:
            root_dir (string): Directory with all the images.
            imagefile_to_class_dictionary (dictionary): Dictionary of image filenames to class for each emotion.
        """
        self.imagefile_to_class_dictionary = imagefile_to_class_dictionary
        self.transform = transforms.Compose([
                                    

        transforms.RandomResizedCrop(96),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
        transforms.Resize((96, 96)),

        
    ])


    def __len__(self):
        return len(self.imagefile_to_class_dictionary.keys())

    def __getitem__(self, idx):
        path_to_image = self.imagefile_to_class_dictionary[idx]['file']
        image = Image.open(path_to_image)
        image = self.transform(image).float()
        label = int(self.imagefile_to_class_dictionary[idx]['label'])
        return image, label


