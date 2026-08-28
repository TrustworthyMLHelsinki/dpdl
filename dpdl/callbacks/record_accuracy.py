import csv
import logging
import os

from ..utils import tensor_to_python_type
from .base_callback import Callback

log = logging.getLogger(__name__)


class RecordAccuracyByEpochCallback(Callback):
    """
    Record train / validation accuracy once per epoch and write the
    values to `<log_dir>/epoch_accuracy.csv`.
    """

    def __init__(self, log_dir: str, metric: str = 'MulticlassAccuracy'):
        super().__init__()

        self.log_dir = log_dir
        self.metric = metric
        self.csv_path = None

        self.train_accuracies = []
        self.val_accuracies = []

    def on_train_start(self, trainer):
        if not self._is_global_zero():
            return

        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_path = os.path.join(self.log_dir, 'epoch_accuracy.csv')

        # write header
        with open(self.csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'train_accuracy', 'val_accuracy'])

    def on_train_epoch_end(self, trainer, epoch, metrics):
        acc = tensor_to_python_type(metrics[self.metric])
        self.train_accuracies.append(acc)

    def on_validation_epoch_end(self, trainer, epoch, metrics, loss):
        acc = tensor_to_python_type(metrics[self.metric])
        self.val_accuracies.append(acc)

    def on_train_end(self, trainer):
        if not self._is_global_zero():
            return

        epochs = len(self.train_accuracies)

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)

            for i in range(epochs):
                train_acc = self.train_accuracies[i]
                val_acc = self.val_accuracies[i] if i < len(self.val_accuracies) else ''

                writer.writerow([i + 1, train_acc, val_acc])

        log.info(f'Per epoch accuracies written to {self.csv_path}')
