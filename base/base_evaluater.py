from abc import ABC


class BaseEvaluater(ABC):
    def __init__(self, config, model, eval_data):
        self.config = config
        self.eval_data   = eval_data
        self.model  = model


    def evaluate(self):
        raise NotImplementedError

    def get_report(self):
        raise NotImplementedError

    def get_confusion_matrix(self):
        raise NotImplementedError



