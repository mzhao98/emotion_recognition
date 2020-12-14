from dependencies import *


class FacialEmotionDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, imagefile_to_class_dictionary, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            imagefile_to_class_dictionary (dictionary): Dictionary of image filenames to class for each emotion.
        """
        self.root_dir = root_dir
        self.imagefile_to_class_dictionary = imagefile_to_class_dictionary

    def __len__(self):
        return sum([len(self.imagefile_to_class_dictionary[key]) for key in self.imagefile_to_class_dictionary])

    def __getitem__(self, idx):
        img_name = self.imagefile_to_class_dictionary[idx]['file']
        image = io.imread(img_name)
        label = self.imagefile_to_class_dictionary[idx]['label']
        return image, label


def plot_data_histogram():
    # Plot Dataset Histograms to visualize distribution.
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

    train_dataset = FacialEmotionDataset(train_dir, train_imagefile_to_class_dictionary)
    test_dataset = FacialEmotionDataset(test_dir, test_imagefile_to_class_dictionary)


    for i in range(3):
        img, label = train_dataset[i]
        plt.imshow(img)
        plt.title('label:'+str(label))
        plt.show()
        plt.close()


    his = np.histogram([train_imagefile_to_class_dictionary[key]['label'] for key in train_imagefile_to_class_dictionary],bins=7)
    fig, ax = plt.subplots()
    offset = .05
    plt.bar(his[1][1:],his[0])
    ax.set_xticks(his[1][1:] + offset)
    ax.set_xticklabels(categories)
    plt.title("Distribution of Class Labels in Training Set")
    plt.show()
    plt.close()

    his = np.histogram([test_imagefile_to_class_dictionary[key]['label'] for key in test_imagefile_to_class_dictionary],bins=7)
    fig, ax = plt.subplots()
    offset = .05
    plt.bar(his[1][1:],his[0])
    ax.set_xticks(his[1][1:] + offset)
    ax.set_xticklabels(categories)
    plt.title("Distribution of Class Labels in Test Set")
    plt.show()
    plt.close()




