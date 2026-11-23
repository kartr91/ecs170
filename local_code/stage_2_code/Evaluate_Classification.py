"""
multiclass classification of digits (weighted/macro)
compare predicted digits w/ actual test digits, then computes and returns
accuracy (% test images predicted the correct digit),
precision (% of the time model predicts a digit and its correct)
recall (% of the correct digits model found)
F1 (combination of precision and recall)
macro: each digit has equal weight
weighted: digits that occur more often in the test set have higher weight
also returns a 10 x 10 confusion matrix
"""

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from local_code.base_class.evaluate import evaluate


class Evaluate_Classification(evaluate):
    def __init__(self):
        super().__init__('classification metrics', '')

    def evaluate(self):
        true_y = self.data['true_y']
        pred_y = self.data['pred_y']

        labels = []
        for digit in range(10): # 0-9
            labels.append(digit)

        metrics = {} # dictionary to store all the metric labels/scores
        metrics['accuracy'] = accuracy_score(true_y, pred_y) # computes accuracy = # correct / # total

        for average in ['macro', 'weighted']:
            scores = precision_recall_fscore_support( # computes precision, recall, F1
                true_y,
                pred_y,
                labels=labels,
                average=average,
                zero_division=0
            )

            metrics['precision_' + average] = scores[0]
            metrics['recall_' + average] = scores[1]
            metrics['f1_' + average] = scores[2]

        # rows are actual digits and columns are predicted digits
        matrix = confusion_matrix(true_y, pred_y, labels=labels)
        metrics['confusion_matrix'] = matrix

        return metrics