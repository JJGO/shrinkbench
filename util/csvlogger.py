import csv
import torch


class CSVLogger:

    def __init__(self, file, columns):
        self.file = open(file, 'w')
        self.columns = columns
        self.values = {}

        self.writer = csv.writer(self.file)
        self.writer.writerow(self.columns)
        self.file.flush()

    def set(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.columns:
                if isinstance(v, torch.Tensor):
                    v = v.item()
                self.values[k] = v
            else:
                raise ValueError(f"{k} not in columns {self.columns}")

    def update(self):
        row = [self.values.get(c, "") for c in self.columns]
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()
