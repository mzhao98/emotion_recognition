from dependencies import *
from dataset import FacialEmotionDataset, FacialEmotionDataset_Augmented, load_dataset_orig
from multistep_models import InitialNet, PositiveNet, NegativeNet


def compute_test_accuracy(model, test_data_loader, test_dataset):
    """
    This function computes test accuracy of a given model.
    Inputs:
        model: (Pytorch model): neural network
        test_data_loader: Test dataset Pytorch dataloader
        test_dataset: Test dataset
    Outputs:
        Test accuracy (float)
    """
  test_accuracy = 0
  for batch_idx, (inputs, labels) in enumerate(test_data_loader):
      inputs = inputs.to(device)
      labels = labels.to(device)

    

      outputs = model(inputs)
      loss = criterion(outputs, labels)

      _, preds = torch.max(outputs, 1)

      test_accuracy += torch.sum(preds == labels.data)

  test_acc = test_accuracy.double() / len(test_dataset)
  return test_acc



def train_model(model, train_dataset, dataloaders, criterion, optimizer, test_data_loader, test_dataset, num_epochs=25):
    """
    This function trains a given model on the datasets provided.
    Inputs:
        model: (Pytorch model): neural network
        train_dataset: Train dataset 
        dataloaders: Training dataloader 
        criterion: Loss function
        optimizer: pytorch optimizer
        test_data_loader: Test dataloader
        test_dataset: Test dataset
        num_epochs: number of training epochs (default 25)

    Outputs:
        model: Trained model 
        train_acc_history: Training accuracies per epoch
        test_acc_history: Test accuracies per epoch
        loss_history: Losses per epoch
    """

    train_acc_history = []
    test_acc_history = []
    loss_history = []

    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()

        curr_loss = 0.0
        curr_correct = 0

        # Iterate over data.
        for batch_idx, (inputs, labels) in enumerate(dataloaders):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            # backward
            loss.backward()
            optimizer.step()

            # statistics
            curr_loss += loss.item() * inputs.size(0)
            curr_correct += torch.sum(preds == labels.data)

        epoch_loss = curr_loss / len(dataloaders)
        loss_history.append(epoch_loss)
        
        epoch_acc = curr_correct.double() / len(train_dataset)

        test_acc = compute_test_accuracy(model, test_data_loader, test_dataset)
            
        train_acc_history.append(epoch_acc)
        test_acc_history.append(test_acc)
        
    return model, train_acc_history, test_acc_history, loss_history


def train_initial_step():
    # This function trains the initial step model of the Multistep CNN.

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = InitialNet()
    model = model.to(device)
    feature_extract = False

    model_params = model.parameters()

    # Create optimizer. All parameters are being optimized
    optimizer = optim.SGD(model_params, lr=0.001, momentum=0.9)


    # Load dataset
    train_dataset = FacialEmotionDataset(train_imagefile_to_class_dictionary)
    train_dataset_augmented = FacialEmotionDataset_Augmented(train_imagefile_to_class_dictionary)

    test_dataset = FacialEmotionDataset(test_imagefile_to_class_dictionary)


    increased_dataset = torch.utils.data.ConcatDataset([train_dataset_augmented,train_dataset])

    train_data_loader = torch.utils.data.DataLoader(increased_dataset, batch_size=32,
                                              shuffle=True,
                                             )
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32,
                                              shuffle=True,
                                             )


    criterion = nn.CrossEntropyLoss()
    num_epochs = 20

    # Train and evaluate
    model, train_acc_history, test_acc_history, loss_history = train_model(model, increased_dataset, train_data_loader, criterion, optimizer, test_data_loader, test_dataset, num_epochs=num_epochs)




def train_positive_network():
    # This function trains the positive step model of the Multistep CNN.

    # Load dataset.
    train_imagefile_to_class_dictionary, test_imagefile_to_class_dictionary = load_positive_dataset()
    train_dataset = FacialEmotionDataset(train_imagefile_to_class_dictionary)
    train_dataset_augmented = FacialEmotionDataset_Augmented(train_imagefile_to_class_dictionary)

    test_dataset = FacialEmotionDataset(test_imagefile_to_class_dictionary)


    increased_dataset = torch.utils.data.ConcatDataset([train_dataset_augmented,train_dataset])

    train_data_loader = torch.utils.data.DataLoader(increased_dataset, batch_size=32,
                                              shuffle=True,
                                             )
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32,
                                              shuffle=True,
                                             )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    positive_model = PositiveNet()
    positive_model = positive_model.to(device)

    positive_model_params = positive_model.parameters()
    
    # Observe that all parameters are being optimized
    optimizer = optim.SGD(positive_model_params, lr=0.001, momentum=0.9)


    criterion = nn.CrossEntropyLoss()
    num_epochs = 20

    # Train and evaluate
    positive_model, positive_train_acc_history, positive_test_acc_history, positive_loss_history = train_model(positive_model, increased_dataset, train_data_loader, criterion, optimizer, test_data_loader, test_dataset, num_epochs=num_epochs)


def train_negative_network():
    # This function trains the negative step model of the Multistep CNN.

    # Load dataset.
    train_imagefile_to_class_dictionary, test_imagefile_to_class_dictionary = load_negative_dataset()
    train_dataset = FacialEmotionDataset(train_imagefile_to_class_dictionary)
    train_dataset_augmented = FacialEmotionDataset_Augmented(train_imagefile_to_class_dictionary)

    test_dataset = FacialEmotionDataset(test_imagefile_to_class_dictionary)


    increased_dataset = torch.utils.data.ConcatDataset([train_dataset_augmented,train_dataset])

    train_data_loader = torch.utils.data.DataLoader(increased_dataset, batch_size=32,
                                              shuffle=True,
                                             )
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32,
                                              shuffle=True,
                                             )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    negative_model = NegativeNet()
    negative_model = negative_model.to(device)

    negative_model_params = negative_model.parameters()
    

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(negative_model_params, lr=0.001, momentum=0.9)


    criterion = nn.CrossEntropyLoss()
    num_epochs = 100

    # Train and evaluate
    negative_model, negative_train_acc_history, negative_test_acc_history, negative_loss_history = train_model(negative_model, increased_dataset, train_data_loader, criterion, optimizer, test_data_loader, test_dataset, num_epochs=num_epochs)


 




def get_multistep_predictions(init_model, positive_model, negative_model):
    """
    This function computes test accuracy of a given model.
    Inputs:
        init_model: (Pytorch model): initial step neural network
        positive_model: (Pytorch model): pos step neural network
        negative_model: (Pytorch model): neg step neural network
        
    Outputs:
        Test accuracy (float)
        output_predictions: List of predictions
    """
    train_imagefile_to_class_dictionary, test_imagefile_to_class_dictionary = load_dataset_orig()

    train_dataset_final = FacialEmotionDataset(train_imagefile_to_class_dictionary)

    test_dataset_final = FacialEmotionDataset(test_imagefile_to_class_dictionary)



    train_data_loader_final = torch.utils.data.DataLoader(train_dataset_final, batch_size=1,
                                              shuffle=True,
                                             )
    test_data_loader_final = torch.utils.data.DataLoader(test_dataset_final, batch_size=1,
                                              shuffle=True,
                                         )


    test_accuracy = 0.0
    output_predictions = []
    for batch_idx, (inputs, labels) in enumerate(test_data_loader_final):
        inputs = inputs.to(device)
        labels = labels.to(device)

      

        outputs = init_model(inputs)
        
        

        _, preds = torch.max(outputs, 1)

        if preds[0] == 1:
          final_preds = [4]

        if preds[0] == 0:
          pos_outputs = positive_model(inputs)
          _, pos_preds = torch.max(pos_outputs, 1)
          if pos_preds[0] == 0:
            final_preds = [0]
          else:
            final_preds = [3]

        if preds[0] == 2:
          neg_outputs = negative_model(inputs)
          _, neg_preds = torch.max(neg_outputs, 1)
          if neg_preds[0] == 0:
            final_preds = [1]
          elif neg_preds[0] == 1:
            final_preds = [2]
          elif neg_preds[0] == 2:
            final_preds = [5]
          else:
            final_preds = [6]

        # statistics
        output_predictions.append(final_preds[0])

        if int(final_preds[0]) == int(labels.data[0]):
          test_accuracy += 1

    test_acc = test_accuracy / len(test_dataset_final)
    
    return test_acc, output_predictions



if __name__ == '__main__':
  train_initial_step()
  train_positive_network()
  train_negative_network()



