# dataset.py

import h5py                        # Library for interacting with HDF5 files
import torch                       # PyTorch library for tensor computations
from torch.utils.data import Dataset  # Base class for all datasets in PyTorch

class MALDISpectrumDataset(Dataset):
    """
    Custom Dataset class for loading MALDI-TOF mass spectra data from an HDF5 file.
    """
    def __init__(self, h5_file_path, transform=None):
        """
        Initialize the dataset.

        Parameters:
        - h5_file_path: str
            Path to the HDF5 file containing the preprocessed data.
        - transform: callable (optional)
            Optional transform to be applied on a sample.
        """
        # Open the HDF5 file in read mode and store the file object
        self.h5_file = h5py.File(h5_file_path, 'r')
        # Store the transform function (if any)
        self.transform = transform

        # Load the mass-to-charge ratio arrays for all spectra
        self.mz = self.h5_file['mz'][:]
        # Load the intensity arrays for all spectra
        self.intensity = self.h5_file['intensity'][:]
        # Load the label indices for all spectra
        self.labels = self.h5_file['labels'][:] 
        
        """ self.transform = transform
        # Open the HDF5 file and load all data into memory
        with h5py.File(h5_file_path, 'r') as h5_file:
            self.mz = h5_file['mz'][:]
            self.intensity = h5_file['intensity'][:]
            self.labels = h5_file['labels'][:]
     """
    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
        - int: Number of samples in the dataset.
        """
        # The total number of samples is the length of the labels array
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset at the specified index.

        Parameters:
        - idx: int
            Index of the sample to retrieve.

        Returns:
        - sample: dict
            A dictionary containing 'mz', 'intensity', and 'label' for the sample.
        """
        # Retrieve the m/z values for the spectrum at index 'idx'
        mz = self.mz[idx]
        # Retrieve the intensity values for the spectrum at index 'idx'
        intensity = self.intensity[idx]
        # Retrieve the label for the spectrum at index 'idx'
        label = self.labels[idx]

        # Create a sample dictionary with m/z, intensity, and label
        sample = {'mz': mz, 'intensity': intensity, 'label': label}

        # If a transform function is provided, apply it to the sample
        if self.transform:
            sample = self.transform(sample)

        # Return the sample
        return sample
