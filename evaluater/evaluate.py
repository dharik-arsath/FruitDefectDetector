import matplotlib.pyplot as plt
from  base.base_evaluater import BaseEvaluater
import numpy as np
from sklearn import metrics


class NotEvaluated(Exception):
    pass

class Evaluater(BaseEvaluater):

    def __init__(self, config, model, eval_data):
        super().__init__(config, model, eval_data)

        self.true_labels = None
        self.predictions = None

    def __get_labels(self):
        true_labels = [label for _, label in self.eval_data]
        return true_labels


    def __is_predicted(self):
        if self.predictions is None:
            return False

        return True

    def __raise_not_predicted(self):
        msg = "You haven't evaluated the data... Please evaluate using .evaluate method and try this method..."
        raise NotEvaluated(msg)


    def evaluate(self):
        pred_prob       = self.model.predict(self.eval_data)
        self.predictions     = np.argmax(pred_prob, axis=1)
        return self.predictions

    def get_report(self):
        if self.__is_predicted() is False:
            self.__raise_not_predicted()
            return

        if self.true_labels is None:
            self.true_labels = self.__get_labels()

        pred_ids    = self.predictions

        return metrics.classification_report(self.true_labels, pred_ids)

    def get_confusion_matrix(self):
        if self.__is_predicted() is False:
            self.__raise_not_predicted()
            return

        if self.true_labels is None:
            self.true_labels = self.__get_labels()

        pred_ids = self.predictions

        return metrics.confusion_matrix(self.true_labels, pred_ids)

#
#
# def plot_acc_vs_valacc(trainer):
#     plt.plot(trainer.acc, label = "training_acc")
#     plt.plot(trainer.val_acc, label = "validation_acc")
#     plt.grid(True)
#     plt.gca().set_ylim(0, 1)
#     plt.legend(fontsize = "x-large")
#     plt.show()
#
# def plot_loss_vs_valloss(trainer):
#     plt.plot(trainer.loss, label="training_loss")
#     plt.plot(trainer.val_loss, label="validation_loss")
#     plt.grid(True)
#     # plt.gca().set_ylim(0, 1)
#     plt.legend(fontsize="x-large")
#     plt.show()
#
