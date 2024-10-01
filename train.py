# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from model import MALDITransformer          # Import the Transformer model defined in model.py
from data_loader import get_data_loaders    # Import the function to get data loaders
from tqdm import tqdm                       # For displaying progress bars
import numpy as np                          # For numerical operations

def train_model(
    h5_file_path,
    num_classes,
    num_epochs=10,
    batch_size=32,
    learning_rate=1e-4
):
    """
    Train the MALDITransformer model.

    Parameters:
    - h5_file_path: str
        Path to the HDF5 file containing the preprocessed data.
    - num_classes: int
        Number of target classes (pathogens to identify).
    - num_epochs: int
        Number of training epochs.
    - batch_size: int
        Number of samples per batch.
    - learning_rate: float
        Learning rate for the optimizer.
    """
    # Determine the device to use (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get the training and validation data loaders
    train_loader, val_loader, class_weights = get_data_loaders(
        h5_file_path,
        batch_size=batch_size
    )

    # Instantiate the model and move it to the selected device
    model = MALDITransformer(num_classes=num_classes).to(device)

    # Define the loss function (Cross-Entropy Loss for classification)
    # criterion = nn.CrossEntropyLoss()

    # Define the loss function (Cross-Entropy Loss for classification) with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Define the optimizer (AdamW optimizer)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Variable to track the best validation accuracy achieved
    best_val_acc = 0.0

    # Training loop over epochs
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        train_losses = []     # List to store training losses
        train_correct = 0     # Counter for correct predictions
        total_samples = 0     # Total number of samples processed

        # Iterate over the training data loader with progress bar
        for mz, intensity, labels in tqdm(
            train_loader,
            desc=f'Epoch {epoch+1}/{num_epochs} - Training'
        ):
            # Move data to the selected device
            mz = mz.to(device)
            intensity = intensity.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass through the model
            outputs = model(mz, intensity)

            # Compute the loss between the model outputs and the true labels
            loss = criterion(outputs, labels)

            # Backward pass (compute gradients)
            loss.backward()

            # Update the model parameters
            optimizer.step()

            # Record the training loss
            train_losses.append(loss.item())

            # Get the predicted classes by taking the argmax over the output logits
            _, preds = torch.max(outputs, 1)

            # Update the count of correct predictions
            train_correct += (preds == labels).sum().item()

            # Update the total number of samples processed
            total_samples += labels.size(0)

        # Calculate the average training accuracy for the epoch
        train_acc = train_correct / total_samples

        # Evaluate the model on the validation set
        val_acc, val_loss = evaluate_model(model, val_loader, criterion, device)

        # Print the training and validation metrics for the current epoch
        print(
            f'Epoch {epoch+1}/{num_epochs}, '
            f'Train Loss: {np.mean(train_losses):.4f}, Train Acc: {train_acc:.4f}, '
            f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}'
        )

        # Check if the validation accuracy is the best achieved so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc  # Update the best validation accuracy
            # Save the model's state dictionary (parameters) to a file
            torch.save(model.state_dict(), 'best_model.pth')

    # Training loop completed
    print(f'Training complete. Best validation accuracy: {best_val_acc:.4f}')

def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate the model on a validation or test dataset.

    Parameters:
    - model: nn.Module
        The trained model to evaluate.
    - data_loader: DataLoader
        DataLoader for the validation or test dataset.
    - criterion: loss function
        Loss function used for evaluation.
    - device: torch.device
        Device to perform computation on (CPU or GPU).

    Returns:
    - val_acc: float
        Validation accuracy.
    - val_loss: float
        Average validation loss.
    """
    model.eval()  # Set the model to evaluation mode
    val_losses = []    # List to store validation losses
    correct = 0        # Counter for correct predictions
    total_samples = 0  # Total number of samples evaluated

    with torch.no_grad():
        # Iterate over the validation data loader with progress bar
        for mz, intensity, labels in tqdm(data_loader, desc='Validation'):
            # Move data to the selected device
            mz = mz.to(device)
            intensity = intensity.to(device)
            labels = labels.to(device)

            # Forward pass through the model (no gradient computation)
            outputs = model(mz, intensity)

            # Compute the loss
            loss = criterion(outputs, labels)

            # Record the validation loss
            val_losses.append(loss.item())

            # Get the predicted classes
            _, preds = torch.max(outputs, 1)

            # Update the count of correct predictions
            correct += (preds == labels).sum().item()

            # Update the total number of samples evaluated
            total_samples += labels.size(0)

    # Calculate the validation accuracy
    val_acc = correct / total_samples

    # Calculate the average validation loss
    val_loss = np.mean(val_losses)

    # Return the validation accuracy and loss
    return val_acc, val_loss

if __name__ == '__main__':
    # Entry point of the script

    h5_file_path = 'preprocessed_data.h5'  # Path to your preprocessed data HDF5 file
    num_classes = 231  # Replace with the actual number of classes in your dataset

    # Call the training function with specified parameters
    train_model(
        h5_file_path,
        num_classes,
        num_epochs=20,
        batch_size=32,
        learning_rate=1e-4
    )
