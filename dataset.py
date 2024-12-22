import numpy as np
import torch
from torch.utils.data import Dataset


class NPYDataset(Dataset):
    def __init__(self, feature_file, label_file):
        """
        Initialize the dataset.
        :param feature_file: .npy file
        :param label_file: .npy file
        """
        # Load features and labels from .npy files
        self.features = feature_file
        self.labels = label_file

        # Ensure the number of samples in features and labels are the same
        assert self.features.shape[0] == self.labels.shape[0], \
            "The number of samples in features and labels must be the same."

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.features)

    def __getitem__(self, idx):
        """
        Retrieve the feature and label at the specified index.
        :param idx: Index of the sample to retrieve.
        :return: A tuple containing the feature as a tensor and the label as a tensor.
        """
        # Convert data from numpy arrays to PyTorch tensors
        feature = torch.from_numpy(self.features[idx]).float()
        label = torch.tensor(self.labels[idx]).float()
        return feature, label



