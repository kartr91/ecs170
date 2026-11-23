"""train and test on the provided stage 3 split"""

from local_code.base_class.setting import setting

# based on setting class
class Setting_Provided_Split(setting):
    def __init__(self):
        super().__init__('provided train/test split', '')

    def load_run_save_evaluate(self):
        # load the data then train and test the model
        self.method.data = self.dataset.load() # calls this dataset's loader, which stores
        # the training/testing arrays
        learned_result = self.method.run() # calls the CNN run method

        # score the test predictions and save the run
        self.evaluate.data = learned_result
        metrics = self.evaluate.evaluate()
        learned_result['metrics'] = metrics
        self.result.data = learned_result
        self.result.save()
        return metrics
