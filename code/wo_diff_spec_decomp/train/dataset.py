import torch
import os
import h5py


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path):
        assert dataset_path.endswith('.h5'), "dataset_path must be the path to a .h5 file"
        assert os.path.exists(dataset_path), "dataset_path is wrong"
        self.dataset_path = dataset_path
        self.dataset = None
        self.dataset_len = 0
        self.keys = []

        self.load_dataset()

    def load_dataset(self):
        with h5py.File(self.dataset_path, 'r') as file:
            self.dataset_len = len(file["aux"])
            name = self.dataset_path[self.dataset_path.rfind('\\')+1:]
        print("Data set %s has %d samples in total" % (name, self.dataset_len))

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(self.dataset_path, 'r')
            self.keys = self.dataset.keys()
        data = {}
        for key in self.keys:
            data[key] = self.dataset[key][index]
        return data


