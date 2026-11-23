'''
Concrete ResultModule class for a specific experiment ResultModule output
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import csv
import os

import matplotlib.pyplot as plt

from local_code.base_class.result import result


class Result_Saver(result):
    def __init__(self, output_folder):
        super().__init__('Stage 2 results', '')
        self.output_folder = output_folder

    def save(self):
        print('saving results...')
        # makes sure our res folder exists
        os.makedirs(self.output_folder, exist_ok=True)

        # save loss/accuracy plots
        self.save_learning_curves()

        # build history.csv - 10 epochs (rows), col1 = epoch #, col2 = training loss, col3 = training accruacy
        # also used to build side-by-side comparison plot later
        history_file = os.path.join(self.output_folder, 'history.csv')
        file = open(history_file, 'w')
        writer = csv.writer(file)
        writer.writerow(['epoch', 'train_loss', 'train_accuracy'])

        for row in self.data['history']:
            epoch = row['epoch']
            loss = row['train_loss']
            accuracy = row['train_accuracy']
            writer.writerow([epoch, loss, accuracy])

        file.close()

        # write the current experiment's settings along with the final test scores to summary.txt
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

        # finally, write the confusion matrix to confusion_matrix.csv
        # actual digits are the rows, predicted digits are columns
        # cell number reflects the count, i.e. row x, column y = how many of digit x are
        # predicted to be y
        matrix_file = os.path.join(self.output_folder, 'confusion_matrix.csv')
        file = open(matrix_file, 'w')
        writer = csv.writer(file)

        labels = []
        for digit in range(10):
            labels.append(digit)
        writer.writerow(['actual / predicted'] + labels)

        matrix = self.data['metrics']['confusion_matrix']
        for digit in range(len(matrix)):
            row = [digit]
            for count in matrix[digit]:
                row.append(count)
            writer.writerow(row)

        file.close()

        print('Results saved to: ' + str(self.output_folder))

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
