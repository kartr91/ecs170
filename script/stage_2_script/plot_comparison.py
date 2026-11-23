"""compare training loss across the five stage 2 experiments """

import csv
import os
import matplotlib.pyplot as plt


results = '../../result/stage_2_result' # storage directory

figure, axes = plt.subplots(1, 2, figsize=(12, 5)) # 1 row, 2 plots, 12x5 size
CE = axes[0]
MSE = axes[1]

runs = { # mapping the names of the folders to the plot labels on the graph
    'baseline': 'Baseline',
    'two_hidden_layers': 'Two hidden layers',
    'learning_rate_001': 'LR 0.01',
    'sgd': 'SGD',
    'mse': 'MSE'
}

for folder in runs:
    label = runs[folder]
    history_file = results + '/' + folder + '/history.csv'

    file = open(history_file)
    rows = list(csv.DictReader(file)) # extracts the list of dictionaries
    file.close()

    epochs = []
    losses = []
    for row in rows:
        epochs.append(int(row['epoch']))
        losses.append(float(row['train_loss']))

    if folder == 'mse':
        MSE.plot(epochs, losses, marker='o', label=label)
    else: # CE
        CE.plot(epochs, losses, marker='o', label=label)

# build the cross entropy plot
CE.set_title('Cross-entropy')
CE.set_xlabel('Epoch')
CE.set_ylabel('Training loss')
CE.set_yscale('log')
CE.legend()

# build the MSE plot
MSE.set_title('Mean squared error')
MSE.set_xlabel('Epoch')
MSE.set_ylabel('Training loss')

figure.tight_layout()

finalcomp_folder = results + '/final_comparison'
os.makedirs(finalcomp_folder, exist_ok=True)

figure.savefig(finalcomp_folder + '/training_comparison.png')
plt.close(figure)