"""make the stage 3 report plots"""

import csv
import os
import matplotlib.pyplot as plt


RESPATH = '../../result/stage_3_result'
FIGSPATH = RESPATH + '/report_figures'


# read the saved epochs and losses from one exp
def read_history(dataset, experiment):
    file = open(RESPATH + '/' + dataset + '/' + experiment + '/history.csv')
    rows = list(csv.DictReader(file))
    file.close()

    epochs = []
    losses = []

    for row in rows:
        epochs.append(int(row['epoch']))
        losses.append(float(row['train_loss']))

    return epochs, losses


# get test accuracy from the saved summary
def read_accuracy(dataset, experiment):
    file = open(RESPATH + '/' + dataset + '/' + experiment + '/summary.txt')

    # find the test accuracy in the summary
    for line in file:
        if line.startswith('accuracy:'):
            parts = line.split(':', 1)
            accuracy = float(parts[1])
            break

    file.close()
    return accuracy

# initial plot to create 3 side-by-side convergence plots
# overall findings:
# MNIST - most learning happens early (sharp drop offs in loss early)
# ORL - loss drops sharply in the middle (epochs 10-30)
# CIFAR - loss falls steadily, indicating that more training would be useful (since
# the plot hasn't flattened out at epoch 10

def plot_convergence():
    # list of tuples, each containing the data set, experiment, and plot title
    # data set and experiment are used to call read_history to get the run data
    plots = [
        ('mnist', 'three_conv_blocks', 'MNIST: 10 epochs'),
        ('orl', 'longer_training', 'ORL: 50 epochs'),
        ('cifar', 'three_conv_blocks', 'CIFAR: 10 epochs')
    ]

    # plot all 3 plots in 1 row side by side
    figure, axes = plt.subplots(1, 3, figsize=(9, 3))

    for index in range(len(plots)):
        run = plots[index] #
        dataset = run[0]
        experiment = run[1]
        title = run[2]
        axis = axes[index]

        epochs, losses = read_history(dataset, experiment)
        axis.plot(epochs, losses, color='#305577') # dark blue
        axis.set_title(title)
        axis.set_xlabel('Epoch')
        axis.set_ylabel('Training loss')
        axis.grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(FIGSPATH + '/convergence.png')
    plt.close(figure)


# 3 side by side plots for depth comparison, following the same structure
# here I compared using 16 and 32filters (2 blocks)
# versus 3 blocks with 16, 32, and 64 filters (3 blocks)
# all blocks used convolution, ReLU, and pooling - see diagram in report
# extra layers allows the CNN to understand more complex patterns

def plot_depth_comparison():
    plots = [
        ('mnist', 'baseline', 'three_conv_blocks', 'MNIST (10 epochs)'),
        ('orl', 'longer_training', 'three_conv_blocks', 'ORL (50 epochs)'),
        ('cifar', 'baseline', 'three_conv_blocks', 'CIFAR (10 epochs)')
    ]

    # 1 row, 3 plots
    figure, axes = plt.subplots(1, 3, figsize=(9, 3))

    # cycle through each plot
    for index in range(len(plots)):
        run = plots[index]
        dataset = run[0]
        first_run = run[1]
        second_run = run[2]
        title = run[3]
        axis = axes[index]

        # get the accuracy for the 2-layer and 3-layer runs
        first_score = read_accuracy(dataset, first_run)
        second_score = read_accuracy(dataset, second_run)
        scores = [first_score, second_score]

        # bar chart
        bars = axis.bar(
            ['2 blocks', '3 blocks'],
            scores,
            color=['#84b2d2', '#305577'] # light blue, dark blue
        )

        score_labels = []
        for score in scores:
            score_labels.append(format(score, '.1%')) # round scores to nearest .1%

        axis.bar_label(bars, labels=score_labels)
        axis.set_title(title)
        axis.set_ylabel('Test accuracy')
        axis.set_ylim(0, 1.08) # extra space for label, value will be 0-100%

    figure.tight_layout()
    figure.savefig(FIGSPATH + '/varyingdepth.png')
    plt.close(figure)

# final convergence plot - in my ablation studies, I noted that 8x8 pooling improved accuracy
# on the ORL dataset from 90->100% (low test set size) and that increasing the number of epochs
# for training the CIFAR set from 10 to 20 was logical because loss was still decreasing steadily
# at epoch 10
def plot_final_convergence():
    selected = [
        ('mnist', 'three_conv_blocks', 'MNIST: 10 epochs'),
        ('orl', 'pool8_50', 'ORL: 50 epochs, 8×8 map'),
        ('cifar', 'epochs_20', 'CIFAR: 20 epochs')
    ]

    figure, axes = plt.subplots(1, 3, figsize=(9, 2.8))

    for index in range(len(selected)):
        run = selected[index]
        dataset = run[0]
        experiment = run[1]
        title = run[2]
        axis = axes[index]

        epochs, losses = read_history(dataset, experiment)
        axis.plot(epochs, losses, color='#305577') # dark blue
        axis.set_title(title)
        axis.set_xlabel('Epoch')
        axis.set_ylabel('Training loss')
        axis.grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(FIGSPATH + '/final_training_convergence.png', dpi=180)
    plt.close(figure)

# final plot is comparing test set accuracy for all three sets according to the
# settings changes in the ablation studies, findings mentioned above
# for MNIST, I added a third layer
# for ORL, increased pooling size from 4x4 to 8x8, increasing spatial detail while also
# increasing the number of parameters by 4x (32*4*4 to 32*8*8)
# for CIFAR, I trained for an additional 10 epochs

def plot_followup_accuracy():
    # setting up labels: format is dataset/original run folder/changed setting folder/
    # plot title/labels for the left and right accuracy bar/accuracy threshold that
    # I was aiming for
    comparisons = [
        (
            'mnist', 'baseline', 'three_conv_blocks',
            'MNIST: model depth', ['2 blocks', '3 blocks'], 0.95
        ),
        (
            'orl', 'longer_training', 'pool8_50',
            'ORL: pooled map', ['4×4', '8×8'], 0.95
        ),
        (
            'cifar', 'three_conv_blocks', 'epochs_20',
            'CIFAR: training length', ['10 epochs', '20 epochs'], 0.70
        )
    ]

    figure, axes = plt.subplots(1, 3, figsize=(9, 3))

    for index in range(len(comparisons)):
        comparison = comparisons[index]
        dataset = comparison[0]
        first_run = comparison[1]
        second_run = comparison[2]
        title = comparison[3]
        labels = comparison[4]
        target = comparison[5]
        axis = axes[index]

        first_score = read_accuracy(dataset, first_run)
        second_score = read_accuracy(dataset, second_run)
        scores = [first_score, second_score]

        bars = axis.bar(
            labels,
            scores,
            color=['#84b2d2', '#305577'] # light/dark blue
        )

        score_labels = []
        for score in scores:
            score_labels.append(format(score, '.2%'))

        axis.bar_label(
            bars,
            labels=score_labels,
            label_type='center',
            color='white'
        )

        # mark the reference accuracy
        axis.axhline(
            target,
            color='#ad493c', # red
            linestyle='--',
            linewidth=1
        )

        # reference line based on the expected thresholds (95% or 70%)
        reference_label = format(target, '.0%') + ' reference'
        axis.text(
            0.98,
            target + 0.08,
            reference_label,
            transform=axis.get_yaxis_transform(),
            ha='right',
            color='#ad493c', # red
            fontsize=8
        )

        axis.set_title(title)
        axis.set_ylabel('Test accuracy')
        axis.set_ylim(0, 1.10)

    figure.tight_layout()
    figure.savefig(FIGSPATH + '/followup_accuracy.png', dpi=180)
    plt.close(figure)


if __name__ == '__main__':
    os.makedirs(FIGSPATH, exist_ok=True)

    plot_convergence()
    plot_depth_comparison()
    plot_final_convergence()
    plot_followup_accuracy()

    print('Report figures saved to: ' + FIGSPATH)