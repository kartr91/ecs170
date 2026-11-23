"""score test predictions and show which classes got mixed up"""

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from local_code.base_class.evaluate import evaluate


class Evaluate_Classification(evaluate):
    def __init__(self, num_classes):
        super().__init__('Stage 3 classification metrics', '')
        self.num_classes = num_classes

    def evaluate(self):
        # gets the actual and predicted arrays
        true_y = self.data['true_y']
        pred_y = self.data['pred_y']

        # include every class (including the ones that the model never predicts)
        labels = []
        for number in range(self.num_classes):
            labels.append(number)

        metrics = {} # dictionary to keep track of all metrics
        metrics['accuracy'] = accuracy_score(true_y, pred_y) # accuracy = correct predictions / total

        # macro treats classes equally, while weighted uses test class size
        for average in ['macro', 'weighted']:
            scores = precision_recall_fscore_support( # calculate precision, recall, and F1
                true_y,
                pred_y,
                labels=labels,
                average=average,
                zero_division=0
            )

            metrics['precision_' + average] = scores[0]
            metrics['recall_' + average] = scores[1]
            metrics['f1_' + average] = scores[2]

        # rows are actual labels and columns are predicted labels
        matrix = confusion_matrix(true_y, pred_y, labels=labels)
        metrics['confusion_matrix'] = matrix

        return metrics
