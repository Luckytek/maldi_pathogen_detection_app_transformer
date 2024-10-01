# data_loader.py

import torch                                     # PyTorch library for tensor computations
from torch.utils.data import DataLoader, Subset          # DataLoader class for creating data loaders
from dataset import MALDISpectrumDataset         # Custom dataset class defined in dataset.py
import h5py
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from collections import Counter

def collate_fn(batch):
    """
    Custom collate function to be used with the DataLoader.

    Parameters:
    - batch: list
        A list of samples fetched from the dataset, where each sample is a dictionary
        containing 'mz', 'intensity', and 'label'.

    Returns:
    - mz_padded: torch.Tensor
        Padded tensor of m/z values with shape (batch_size, max_length).
    - intensity_padded: torch.Tensor
        Padded tensor of intensity values with shape (batch_size, max_length).
    - labels: torch.Tensor
        Tensor of labels with shape (batch_size,).
    """
    batch_size = len(batch)  # Number of samples in the batch

    # Find the maximum sequence length in the batch
    max_length = max([len(sample['intensity']) for sample in batch])

    # Initialize tensors for m/z and intensity with zeros, padded to the max_length
    mz_padded = torch.zeros((batch_size, max_length), dtype=torch.float32)
    intensity_padded = torch.zeros((batch_size, max_length), dtype=torch.float32)

    # Initialize a tensor for labels
    labels = torch.tensor([sample['label'] for sample in batch], dtype=torch.long)

    # Loop over each sample in the batch to populate the tensors
    for i, sample in enumerate(batch):
        length = len(sample['intensity'])  # Length of the current sequence

        # Copy the m/z values into the padded tensor
        mz_padded[i, :length] = torch.tensor(sample['mz'], dtype=torch.float32)

        # Copy the intensity values into the padded tensor
        intensity_padded[i, :length] = torch.tensor(sample['intensity'], dtype=torch.float32)

    # Return the padded tensors and labels
    return mz_padded, intensity_padded, labels

# def get_data_loaders(h5_file_path, batch_size=4, num_workers=0):
        
    """
    Create data loaders for training and validation datasets.

    Parameters:
    - h5_file_path: str
        Path to the HDF5 file containing the preprocessed data.
    - batch_size: int
        Number of samples per batch.
    - num_workers: int
        Number of subprocesses to use for data loading.

    Returns:
    - train_loader: DataLoader
        DataLoader for the training dataset.
    - val_loader: DataLoader
        DataLoader for the validation dataset.
    """
    """ # Instantiate the custom dataset
    dataset = MALDISpectrumDataset(h5_file_path)

    # Determine the total number of samples in the dataset
    total_samples = len(dataset)

    # Calculate the sizes of the training and validation sets (80% train, 20% validation)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size

    # Split the dataset into training and validation subsets
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Set seed for reproducibility
    )

    # Create a DataLoader for the training dataset
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,                 # Shuffle data at every epoch
        num_workers=num_workers,
        collate_fn=collate_fn         # Use the custom collate function
    )

    # Create a DataLoader for the validation dataset
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,                # Do not shuffle validation data
        num_workers=num_workers,
        collate_fn=collate_fn         # Use the custom collate function
    )

    # Return the DataLoaders
    return train_loader, val_loader """

def get_data_loaders(h5_file_path, batch_size=32, num_workers=0):
    """
    Creates training and validation DataLoaders from an HDF5 file after removing singleton classes.

    Args:
        h5_file_path (str): Path to the HDF5 file containing 'spectra' and 'labels' datasets.
        batch_size (int, optional): Number of samples per batch. Defaults to 4.
        num_workers (int, optional): Number of subprocesses for data loading. Defaults to 0.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Load labels from HDF5 file
    with h5py.File(h5_file_path, 'r') as h5f:
        labels = h5f['labels'][:]

    labels = np.array(labels)
    label_counts = Counter(labels)
    print("Original Class Distribution:", label_counts)

    # Identify singleton classes (classes with only one instance)
    singleton_classes = [cls for cls, count in label_counts.items() if count == 1]
    if singleton_classes:
        print(f"Singleton Classes (only 1 instance): {singleton_classes}")
    else:
        print("No singleton classes detected.")

    # If there are singleton classes, remove them from the dataset
    if singleton_classes:
        # Identify indices of samples that are NOT in singleton classes
        non_singleton_indices = np.where(~np.isin(labels, singleton_classes))[0]
        filtered_labels = labels[non_singleton_indices]
    else:
        # If no singleton classes, use all indices
        non_singleton_indices = np.arange(len(labels))
        filtered_labels = labels

    # Perform stratified splitting on the filtered data
    try:
        train_indices, val_indices = train_test_split(
            non_singleton_indices,
            test_size=0.2,
            stratify=filtered_labels,
            random_state=42
        )
    except ValueError as e:
        print("Error during train_test_split:", e)
        print("Ensure that all classes have at least two instances after removing singletons.")
        raise e

    # Create the full dataset
    full_dataset = MALDISpectrumDataset(h5_file_path)

    # Create training and validation subsets using the filtered indices
    train_subset = Subset(full_dataset, indices=train_indices)
    val_subset = Subset(full_dataset, indices=val_indices)

    # Compute class weights based on the training data
    train_labels = labels[train_indices]
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    
    # Create DataLoaders with appropriate settings
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle for training
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for validation
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    # Display new class distributions
    new_train_labels = labels[train_indices]
    new_val_labels = labels[val_indices]
    print("New Training Class Distribution:", Counter(new_train_labels))
    print("New Validation Class Distribution:", Counter(new_val_labels))

    return train_loader, val_loader, class_weights