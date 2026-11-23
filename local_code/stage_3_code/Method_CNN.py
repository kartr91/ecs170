"""One class to handle all three datasets
with a convolutional neural net"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from local_code.base_class.method import method

# based on pytorch's neural net, and follows the abstract class functionality in method.py
class Method_CNN(method, nn.Module):
    # input channels and number of classes vary based on the dataset being used
    # see script_cnn.py for info on the parameters
    def __init__(self, input_channels, num_classes, conv_channels=(16, 32),
                 kernel_size=3, padding=1, stride=1, pool_type='max',
                 hidden_size=128, learning_rate=0.001, max_epoch=10,
                 batch_size=128, optimizer='adam', loss_function='cross_entropy',
                 pooled_size=4):
        # initialize the base class and pytorch's
        method.__init__(self, 'Stage 3 CNN')
        nn.Module.__init__(self)

        # keep these settings with the saved results
        self.config = {
            'input_channels': input_channels,
            'num_classes': num_classes,
            'conv_channels': list(conv_channels),
            'kernel_size': kernel_size,
            'padding': padding,
            'stride': stride,
            'pool_type': pool_type,
            'hidden_size': hidden_size,
            'learning_rate': learning_rate,
            'max_epoch': max_epoch,
            'batch_size': batch_size,
            'optimizer': optimizer,
            'loss': loss_function,
            'pooled_size': pooled_size,
        }

        # choose the loss function for this run based on input
        if loss_function == 'cross_entropy':
            self.loss_function = nn.CrossEntropyLoss()
        elif loss_function == 'mse':
            self.loss_function = nn.MSELoss()

        layers = [] # list to hold the network's layers in order
        previous_channels = input_channels # 1 for MINST/ORL, 3 for CIFAR

        # select max/avg pooling based on input
        if pool_type == 'max':
            pooling = nn.MaxPool2d
        elif pool_type == 'average':
            pooling = nn.AvgPool2d

        # build 1 block per channel, e.g. (16,32) builds 2 blocks
        for channels in conv_channels:
            convolution = nn.Conv2d(
                previous_channels, # number of incoming maps to read
                channels, # number of filters + output maps to create
                kernel_size, # size of filter window
                stride=stride, # how far the filter moves each step
                padding=padding # padding of zero's  around the border
            )

            layers.append(convolution) # add the convolution layer to the network
            layers.append(nn.ReLU()) # add ReLU, turning negative values to 0
            layers.append(pooling(2)) # uses earlier pooling type with a 2x2 window ->
            # feature maps height and width are cut in half

            previous_channels = channels # keep track of # of feature maps created by this block
            # so next convolution accepts the same number

        layers.append(nn.AdaptiveAvgPool2d((pooled_size, pooled_size))) # feature maps become 4x4
        layers.append(nn.Flatten()) # turns the maps into a single list, eg 32*4*4->512 values

        num_features = previous_channels * pooled_size * pooled_size # # of values produced by the flattening
        layers.append(nn.Linear(num_features, hidden_size)) # combine the flattened features into 128 values
        layers.append(nn.ReLU()) # convert negative values to 0
        layers.append(nn.Linear(hidden_size, num_classes)) # finally convert those 128 values into 10 or 40 output
        # classes based on dataset

        self.network = nn.Sequential(*layers)
        # make each feature map the same size before the dense layers
        layers.extend([
            nn.AdaptiveAvgPool2d((pooled_size, pooled_size)),
            nn.Flatten(),
            nn.Linear(previous_channels * pooled_size * pooled_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        ])
        self.network = nn.Sequential(*layers) # put all the layers into the network in order

    def forward(self, images):
        return self.network(images) # runs the images through every layer in the network
        # and returns one raw score per class

    def fit(self, X, y):
        # convert np arrays -> pytorch tensors
        images = torch.from_numpy(X)
        labels = torch.from_numpy(y)

        training_data = TensorDataset(images, labels) # pair images and labels

        loader = DataLoader(
            training_data,
            batch_size=self.config['batch_size'], # batch size which was defined earlier
            shuffle=True # shuffles the order of batches per epoch
        )
        # choose the optimizer to determine how the weights are updated
        if self.config['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.config['learning_rate']
            )
        elif self.config['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.config['learning_rate']
            )

        # track loss and correct answers across each epoch
        history = []
        self.train() # training mode

        for epoch in range(1, self.config['max_epoch'] + 1): # train for the defined # of epochs
            # reset totals each epoch
            total_loss = 0.0
            correct = 0

            for images, labels in loader: # cycle through all batches
                optimizer.zero_grad() # clear prev gradients
                scores = self(images) # runs through forward() to get raw scores

                if self.config['loss'] == 'mse':
                    probabilities = torch.softmax(scores, dim=1) # use softmax to convert raw scores to 0-1 probability

                    targets = nn.functional.one_hot(
                        labels,
                        num_classes=self.config['num_classes']
                    ) # converts correct labels into a list with 1 for correct class, 0 otherwise
                    targets = targets.float() # convert to decimal values to be comparable w/ probabilities
                    loss = self.loss_function(probabilities, targets) # MSE
                else:
                    loss = self.loss_function(scores, labels) # directly calculate loss for CE

                loss.backward() # backprop to calculate gradients
                optimizer.step() # update weights using these gradients and prev parameters (learning rate, etc)

                # get # of images + average loss for this batch
                batch_size = len(labels)
                batch_loss = loss.item() # avg loss
                total_loss += batch_loss * batch_size #weighted loss by batch size
                # (to account for final batch being possibly smaller)
                predictions = scores.detach().argmax(dim=1) # pick highest scoring class
                batch_correct = 0
                for index in range(len(labels)): # count how many images were classified correctly
                    if predictions[index] == labels[index]:
                        batch_correct += 1

                correct += batch_correct

            average_loss = total_loss / len(y)
            accuracy = correct / len(y)

            row = {
                'epoch': epoch,
                'train_loss': average_loss,
                'train_accuracy': accuracy,
            }
            history.append(row)

            accuracy_percent = accuracy * 100
            print(
                'Epoch:', epoch,
                'Loss:', round(average_loss, 4),
                'Training accuracy:', round(accuracy_percent, 2), '%'
            )

        return history

    # test in batches without tracking gradients
    def test(self, X):
        self.eval() # evaluation mode
        predictions = []
        batch_size = self.config['batch_size']

        # loop through the test images in batches (step size is batch size)
        for start in range(0, len(X), batch_size):
            end = start + batch_size # end index
            images = torch.from_numpy(X[start:end]) # converts current batch to tensor

            scores = self(images) # runs batch through CNN
            predicted_labels = scores.argmax(dim=1) # picks highest scoring class for each image
            predictions.append(predicted_labels)

        predictions = torch.cat(predictions) # combines batches into 1 tensor
        return predictions.numpy() # convert to np array for evaluator

    def run(self):
        # train, test, and pass the results
        history = self.fit(self.data['train']['X'], self.data['train']['y'])
        predictions = self.test(self.data['test']['X'])
        return {
            'pred_y': predictions,
            'true_y': self.data['test']['y'],
            'history': history,
            'config': self.config,
        }
