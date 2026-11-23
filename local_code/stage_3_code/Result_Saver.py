"""save the plots settings and test scores for one cnn run
similar to the result saver for stage 2, but handles
dataset-specific differences
"""

import csv
import os

import matplotlib.pyplot as plt

from local_code.base_class.result import result

class Result_Saver(result):
    def __init__(self, output_folder):\
        # calls the abstract class base_class's constructor
        super().__init__('The results for stage 3 are: ', '')
        self.output_folder = output_folder

    def save(self):
        # make the folder and save the training plots
        os.makedirs(self.output_folder, exist_ok=True)
        # makes the loss/accuracy plots - see below
        self.save_learning_curves()

        # store each epoch stats so runs can be compared later if needed
        history_file = os.path.join(self.output_folder, 'history.csv')
        file = open(history_file, 'w')
        writer = csv.writer(file)
        writer.writerow(['epoch', 'train_loss', 'train_accuracy'])
        for row in self.data['history']:
            writer.writerow([row['epoch'], row['train_loss'], row['train_accuracy']])
        file.close()

        # put the settings and test scores in one text file
        summary_file = os.path.join(self.output_folder, 'summary.txt')
        file = open(summary_file, 'w')
        file.write('Settings\n')
        for name in self.data['config']:
            value = self.data['config'][name]
            file.write(name + ': ' + str(value) + '\n')
        file.write('\nTest scores\n')
        for name in self.data['metrics']:
            if name != 'confusion_matrix':
                value = self.data['metrics'][name]
                score = format(value, '.6f')
                file.write(name + ': ' + score + '\n')
        file.close()

        # save confusion matrix as CSV
        # for the ORL dataset, labels start at 1 but model labels start at 0
        label_offset = self.data['config'].get('offset', 0)
        # 10 for MNIST/CIFAR, 40 for ORL
        num_classes = self.data['config']['num_classes']
        # 0-9 for MNIST/CIFAR, 1-40 for ORL
        labels = []
        for number in range(num_classes):
            labels.append(number + label_offset)

        # gets the matrix calculated by the evaluator earlier
        matrix = self.data['metrics']['confusion_matrix']
        matrix_file = os.path.join(self.output_folder, 'confusion_matrix.csv')

        # write the matrix to file
        file = open(matrix_file, 'w')
        writer = csv.writer(file)
        writer.writerow(['actual / predicted'] + labels)

        for index in range(num_classes):
            row = [labels[index]]
            for count in matrix[index]:
                row.append(count)
            writer.writerow(row)

        file.close()

        print(f'Results saved to: {self.output_folder}')

    # makes two side by side plots in the output folder
    # left plot - epoch vs loss, right - epoch vs accuracy
    def save_learning_curves(self):
        # gets the recorded data from training (saved earlier)
        # entries are dict entries, e.g.
        # {'epoch': 1, 'train_loss': 0.57, 'train_accuracy': 0.81}
        history = self.data['history']

        #make three lists - epochs for the x axis and losses/accuracies for the y-axis for
        # both plots from the history
        epochs = []
        losses = []
        accuracies = []
        for row in history:
            epochs.append(row['epoch'])
            losses.append(row['train_loss'])
            accuracies.append(row['train_accuracy'])

        # create a figure skeleton with 1 row and 2 columns for 2 plots
        figure, (loss_chart, accuracy_chart) = plt.subplots(1, 2, figsize=(10, 4))

        # left plot - epoch vs loss - o places markers at each ordered pair on the plot
        loss_chart.plot(epochs, losses, marker='o')
        loss_chart.set(xlabel='Epoch', ylabel='Training loss', title='Training loss')

        # same for accuracy, constricting the range to [0, 1]
        accuracy_chart.plot(epochs, accuracies, marker='o')
        accuracy_chart.set(
            xlabel='Epoch', ylabel='Training accuracy', title='Training accuracy', ylim=(0, 1),
        )
        # save both plots in one image in the specific output folder
        figure.tight_layout()
        figure.savefig(os.path.join(self.output_folder, 'learning_curves.png'))
        plt.close(figure)
