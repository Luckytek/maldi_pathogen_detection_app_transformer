# test_model.py

import h5py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time
import collections

from model import MALDITransformer  # Import your Transformer model
import argparse

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Test MALDI Transformer Model')
    parser.add_argument('--h5_file', type=str, default='preprocessed_data.h5', help='Path to the HDF5 file')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='Path to the trained model file')
    args = parser.parse_args()

    h5_file_path = args.h5_file
    model_path = args.model_path

    print("Loading preprocessed data...")
    # Load the preprocessed data
    with h5py.File(h5_file_path, 'r') as h5f:
        mz_data = h5f['mz'][:]
        intensity_data = h5f['intensity'][:]
        labels = h5f['labels'][:]
        species_labels = h5f['species_labels'][:]

    # Decode species labels if they are stored as bytes
    if isinstance(species_labels[0], bytes):
        species_labels = [label.decode('utf-8') for label in species_labels]

    # Map label indices to species names
    label_to_species = {i: species_labels[i] for i in range(len(species_labels))}
    print(f"Species labels mapping: {label_to_species}")

    # Check the distribution of labels in the dataset
    unique_labels = set(labels)
    print(f"Unique labels in the dataset: {unique_labels}")

    label_counts = collections.Counter(labels)
    print("Label counts in the dataset:")
    for label, count in label_counts.items():
        species_name = label_to_species.get(label, "Unknown")
        print(f"Label {label} ({species_name}): {count}")

    # Display available spectra indices
    num_spectra = len(labels)
    print(f"Total number of spectra: {num_spectra}")
    print(f"Available indices: 0 to {num_spectra - 1}")

    # Prompt the user to select a spectrum index
    spectrum_index = int(input(f"Enter the index of the spectrum to analyze (0 to {num_spectra - 1}): "))
    if spectrum_index < 0 or spectrum_index >= num_spectra:
        print("Invalid index. Exiting.")
        return

    print("Loading the selected spectrum...")
    # Get the selected spectrum and label
    mz = mz_data[spectrum_index]
    intensity = intensity_data[spectrum_index]
    true_label = labels[spectrum_index]
    true_species = label_to_species[true_label]

    print(f"True label for index {spectrum_index}: {true_label}")
    print(f"True species for index {spectrum_index}: {true_species}")
    print(f"Sequence length before preprocessing: {len(intensity)}")

    print("Displaying the spectrum plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(mz, intensity)
    plt.xlabel('m/z')
    plt.ylabel('Intensity')
    plt.title(f'Spectrum Index: {spectrum_index}, True Species: {true_species}')
    plt.show(block=False)
    plt.pause(1)  # Pause to ensure the plot displays
    print("Spectrum plot displayed. Proceeding to preprocess the spectrum...")

    print("Starting spectrum preprocessing...")
    # Preprocess the spectrum
    mz_tensor = torch.tensor(mz, dtype=torch.float32).unsqueeze(0)  # Shape: (1, seq_length)
    intensity_tensor = torch.tensor(intensity, dtype=torch.float32).unsqueeze(0)  # Shape: (1, seq_length)

    # Pad or truncate the sequences to a fixed length
    max_seq_length = 500  # Adjust as needed
    if len(intensity) > max_seq_length:
        print("Truncating the sequence...")
        # Truncate the sequence
        indices = np.linspace(0, len(intensity) - 1, num=max_seq_length, dtype=int)
        mz_tensor = mz_tensor[:, indices]
        intensity_tensor = intensity_tensor[:, indices]
    elif len(intensity) < max_seq_length:
        print("Padding the sequence...")
        # Pad the sequence
        pad_length = max_seq_length - len(intensity)
        mz_tensor = torch.nn.functional.pad(mz_tensor, (0, pad_length))
        intensity_tensor = torch.nn.functional.pad(intensity_tensor, (0, pad_length))

    print(f"mz_tensor shape: {mz_tensor.shape}")
    print(f"intensity_tensor shape: {intensity_tensor.shape}")

    # Transpose tensors to match model input shape (seq_length, batch_size)
    # If your model uses batch_first=True, you may need to adjust the dimensions accordingly
    mz_tensor = mz_tensor.transpose(0, 1)  # Shape: (seq_length, batch_size)
    intensity_tensor = intensity_tensor.transpose(0, 1)  # Shape: (seq_length, batch_size)

    print("Spectrum preprocessing completed.")

    print("Loading the trained model...")
    num_classes = len(species_labels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = MALDITransformer(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # Move data to the device
    mz_tensor = mz_tensor.to(device)
    intensity_tensor = intensity_tensor.to(device)

    print(f"Model is on device: {next(model.parameters()).device}")
    print(f"mz_tensor is on device: {mz_tensor.device}")
    print(f"intensity_tensor is on device: {intensity_tensor.device}")

    print("Testing model with dummy input...")
    dummy_mz_tensor = torch.randn(max_seq_length, 1).to(device)
    dummy_intensity_tensor = torch.randn(max_seq_length, 1).to(device)
    try:
        outputs = model(dummy_mz_tensor, dummy_intensity_tensor)
        print("Model test with dummy input completed successfully.")
    except Exception as e:
        print(f"An error occurred during model testing: {e}")
        return

    print("Making the prediction...")
    start_time = time.time()
    # Make the prediction
    with torch.no_grad():
        try:
            outputs = model(mz_tensor, intensity_tensor)
            print(f"Model raw outputs before aggregation: {outputs}")

            # Aggregate the outputs over the sequence length
            outputs = outputs.mean(dim=0)  # Shape: [batch_size, num_classes]
            print(f"Outputs after mean over sequence: {outputs}")

            # Remove batch dimension if batch_size = 1
            outputs = outputs.squeeze(0)  # Shape: [num_classes]
            print(f"Outputs after squeezing batch dimension: {outputs}")

            # Get the predicted class
            _, predicted_label = torch.max(outputs, dim=0)
            predicted_label = predicted_label.item()
        except Exception as e:
            print(f"An error occurred during model prediction: {e}")
            return
    end_time = time.time()
    print(f"Prediction completed in {end_time - start_time:.2f} seconds.")

    predicted_species = label_to_species.get(predicted_label, "Unknown")

    # Display the predicted pathogen
    print(f"Predicted Species Label: {predicted_label}")
    print(f"Predicted Species: {predicted_species}")
    print(f"True Species Label: {true_label}")
    print(f"True Species: {true_species}")

    if predicted_species == true_species:
        print("Prediction is correct!")
    else:
        print("Prediction is incorrect.")

if __name__ == '__main__':
    main()
