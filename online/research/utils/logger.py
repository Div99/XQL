import os
import csv
from abc import ABC, abstractmethod
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except:
    pass

class Writer(ABC):

    def __init__(self, path, on_eval=False):
        self.path = path
        self.on_eval= on_eval
        self.values = {}

    def record(self, key, value):
        self.values[key] = value
    
    def dump(self, step, eval=False):
        if not self.on_eval or eval:
            self._dump(step)

    @abstractmethod
    def _dump(self, step, eval=False):
        return NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError

class TensorBoardWriter(Writer):

    def __init__(self, path, on_eval=False):
        super().__init__(path, on_eval=on_eval)
        self.writer = SummaryWriter(self.path)

    def _dump(self, step):
        for k in self.values.keys():
            self.writer.add_scalar(k, self.values[k], step)
        self.writer.flush()
        self.values.clear()

    def close(self):
        self.writer.close()

class CSVWriter(Writer):

    def __init__(self, path, on_eval=True):
        super().__init__(path, on_eval=on_eval)
        self._csv_file_handler = None
        self.csv_logger = None
        self.num_keys = 0
        
    def _reset_csv_handler(self):
        if self._csv_file_handler is not None:
            self._csv_file_handler.close() # Close our fds
        self.csv_file_handler = open(os.path.join(self.path, "log.csv"), "w")
        self.csv_logger = csv.DictWriter(self.csv_file_handler, fieldnames=list(self.values.keys()))
        self.csv_logger.writeheader()

    def _dump(self, step):
        # Record the step
        self.values["step"] = step
        if len(self.values) > self.num_keys:
            # Got a new key, so re-create the writer
            self.num_keys = len(self.values)
            self._reset_csv_handler()
        # We should now have all the keys
        self.csv_logger.writerow(self.values)
        self.csv_file_handler.flush()
        # Note: Don't reset the CSV because the file handler doesn't support it.

    def close(self):
        self.csv_file_handler.close()

class WandBWriter(Writer):

    def __init__(self, path, on_eval=True):
        super().__init__(path, on_eval=on_eval)
        # No extra init steps, just mark eval as True

    def _dump(self, step):
        wandb.log(self.values, step=step)
        self.values.clear() # reset the values

    def close(self):
        wandb.finish()

class Logger(object):

    def __init__(self, path, writers=['tb', 'csv']):
        self.writers = []
        for writer in writers:
            self.writers.append(
                {
                    'tb': TensorBoardWriter,
                    'csv': CSVWriter,
                    'wandb': WandBWriter
                }[writer](path)
            )

    def record(self, key, value):
        for writer in self.writers:
            writer.record(key, value)

    def dump(self, step, eval=False):
        for writer in self.writers:
            writer.dump(step, eval=eval)

    def close(self):
        for writer in self.writers:
            writer.close()
