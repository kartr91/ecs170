"""For labeling movie reviews positive/negative
Uses the same classification metrics as Stage 3
"""

from local_code.stage_3_code.Evaluate_Classification import Evaluate_Classification as Stage3Evaluator


class Evaluate_Classification(Stage3Evaluator):
    def __init__(self, num_classes=2):
        super().__init__(num_classes)
        self.evaluate_name = "Stage 4 sentiment classification metrics"
