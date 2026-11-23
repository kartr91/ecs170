"""Connect prepared data, the model, evaluator, and result saver as in Stage 3."""

from local_code.base_class.setting import setting


class Setting_Provided_Split(setting):
    def __init__(self):
        super().__init__("Stage 4 train/validation/test", "Best validation-loss checkpoint")

    def load_run_save_evaluate(self):
        # The runner prepares data once so all recurrent units get identical splits.
        self.method.data = self.dataset
        learned = self.method.run()
        if self.evaluate is not None:
            self.evaluate.data = learned
            learned["metrics"].update(self.evaluate.evaluate())
        self.result.data = learned
        self.result.model = self.method
        self.result.save()
        return learned
