'''
Multi-layer perceptron modelling
        X: training images
        y: correct digit labels
'''
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from local_code.base_class.method import method


class Method_MLP(method, nn.Module):
    """Set up the model with the following default characteristics:
        -1 hidden layer, 128 neurons
        -Learning rate: 0.001
        -Max epochs of 10
        -Batch size b=128
        -Optimizer: adam
        -Loss: CE
    """
    def __init__(self, hidden_sizes=(128,), learning_rate=0.001,
                 epochs=10, batch_size=128, opt='adam',
                 loss_name='cross_entropy'):
        method.__init__(self, "MLP")
        nn.Module.__init__(self)
        # loss functions used: CE or MSE
        losstype = {
            'cross_entropy': nn.CrossEntropyLoss,
            'mse': nn.MSELoss,
        }
        # picks loss function from losstype based on input
        self.loss_function = losstype[loss_name]()
        #input size of 784 (28x28 pixels), other characteristics - see above
        self.config = {
            'input_size': 784,
            'hidden_sizes': list(hidden_sizes),
            'output_size': 10,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'batch_size': batch_size,
            'optimizer': opt,
            'loss': type(self.loss_function).__name__,
        }
        layers = [] # holds network pieces
        previous_size = 784  # number to track how many numbers from previous layer
        for hidden_size in hidden_sizes: # cycles through hidden layers: most experiments had 1 layer, 1 of them had 2 (784->128->128->10)
            # linear takes the 784 pixel values and converts it to 128 values
            # ReLU turns the negative values and converts them to 0 (activation function)
            layers.extend([nn.Linear(previous_size, hidden_size), nn.ReLU()])
            previous_size = hidden_size
        layers.append(nn.Linear(previous_size, 10)) # 128 neurons -> 10 digits 0-9
        self.network = nn.Sequential(*layers) # builds network by storing layers in sequential order

    """
    Takes a batch of images with 784 pixels representing a 28 by 28 image, 
    passes them through the network layers 
    and returns 10 scores for each image (labels 0-9).
    The highest score is the prediction"""
    def forward(self, x):
        return self.network(x)

    """
    Trains the multilayer perceptron 
    X: training images
    y: correct digit labels
    """
    def fit(self, X, y):
        # pair images w/ labels, then shuffle the training batches
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
            batch_size=self.config['batch_size'], shuffle=True,
        )
        # choose the experiment-specific optimizer from Adam/row-stochastic GD (SGD)
        optimizer_classes = {'adam': torch.optim.Adam, 'sgd': torch.optim.SGD}
        optimizer_class = optimizer_classes[self.config['optimizer']]
        # update model params using the learning rate from config
        optimizer = optimizer_class(self.parameters(), lr=self.config['learning_rate'])
        # history stores per-epoch results for plotting purposes
        history = []
        # put model in training mode
        self.train()
        # run for the number of epochs specified in config
        for epoch in range(1, self.config['epochs'] + 1):
            # reset loss/correct-prediction total at the beginning of each epoch
            total_loss = 0.0
            correct = 0
            # cycles through dataset, 1 batch at a time
            for features, labels in loader:
                # reset gradients, then get a score (logit) for each digit
                optimizer.zero_grad()
                # run features through model:
                scores = self(features)
                # pick loss function - either mean squared error or cross entropy
                if self.config['loss'] == 'MSELoss':
                    # MSE: softmax converts these scores to probabilities over
                    # the range [0, 1] (summing to probability of 1)
                    probabilities = torch.softmax(scores, dim=1)

                    targets = nn.functional.one_hot(labels, num_classes=10).float()
                    loss = self.loss_function(probabilities, targets)
                else:
                    #
                    loss = self.loss_function(scores, labels)
                #calculate how much each param contributed to the loss
                loss.backward()
                # use these gradients to update the parameters once per batch
                optimizer.step()
                #add this batch's loss to this epoch's total
                total_loss += loss.item() * len(labels)
                # picks the highest scoring digit for each example, then counts correect predictions
                correct += (scores.detach().argmax(dim=1) == labels).sum().item()
            # format a row to print to output
            row = {
                'epoch': epoch,
                'train_loss': total_loss / len(y),
                'train_accuracy': correct / len(y),
            }
            history.append(row)
            print(f"Epoch {epoch:2d}: loss={row['train_loss']:.4f}, "
                  f"training accuracy={row['train_accuracy']:.4%}")
        return history

    def test(self, X):
        self.eval() # evaluation mode
        predictions = [] # list of per-batch answers
        for start in range(0, len(X), self.config['batch_size']):
            features = torch.from_numpy(X[start:start + self.config['batch_size']])
            predictions.append(self(features).argmax(dim=1)) # picks the class with the highest score per image
        return torch.cat(predictions).numpy()

    def run(self):
        # train using training features X, correct labels y
        # contains loss/accuracy for each epoch
        history = self.fit(self.data['train']['X'], self.data['train']['y'])
        # pass the testing data for evaluation
        predictions = self.test(self.data['test']['X'])
        return {
            'pred_y': predictions,
            'true_y': self.data['test']['y'],
            'history': history,
            'config': self.config,
        }
