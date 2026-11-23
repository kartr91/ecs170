"""
for each experiment:
1) loads train/test files
2) passes them to MLP
3) MLP trains/predicts test digits
4) give predictions (learned_result) to evaluator
5) passes scores and training history to saver (for building plots later)
6) returns scores (to be printed)
"""

from local_code.base_class.setting import setting


class Setting_Provided_Split(setting):
    def __init__(self):
        super().__init__('provided train/test split', '')

    def load_run_save_evaluate(self):
        self.method.data = self.dataset.load()
        learned_result = self.method.run()
        self.evaluate.data = learned_result
        # pass scores to saver
        metrics = self.evaluate.evaluate()
        learned_result['metrics'] = metrics
        self.result.data = learned_result
        self.result.save()
        return metrics
