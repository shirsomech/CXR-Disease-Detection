from CXRDataset import CXRDataset
import torch
from config import BATCH_SIZE, NUM_WORKERS

class DatasetManager(object):
    def __init__(self, training_dataset: CXRDataset, testing_dataset: CXRDataset):
        self._datasets = {
            "Train": training_dataset,
            "Val": testing_dataset
        }
        self._dataloaders = {
            "Train": torch.utils.data.DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS),
            "Val": torch.utils.data.DataLoader(testing_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        }
        self._dataset_sizes = {key: len(value) for key, value in self._datasets.items()}

    @property
    def datasets(self):
        return self._datasets

    @property
    def dataset_sizes(self):
        return self._dataset_sizes

    @property
    def dataloaders(self):
        return self._dataloaders

        