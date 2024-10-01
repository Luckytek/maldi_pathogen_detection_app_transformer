# model.py
import torch
import torch.nn as nn
import numpy as np

class MALDITransformer(nn.Module):
    """
    Transformer model for identifying pathogens from MALDI-TOF mass spectra.
    """
    def __init__(
        self,
        num_classes,
        d_model=128,
        nhead=8,
        num_layers=6,
        dim_feedforward=512,
        dropout=0.1
    ):
        """
        Initialize the MALDITransformer model.

        Parameters:
        - num_classes: int
            Number of target classes (pathogens to identify).
        - d_model: int
            Dimensionality of the embedding space.
        - nhead: int
            Number of attention heads in the Transformer encoder.
        - num_layers: int
            Number of Transformer encoder layers.
        - dim_feedforward: int
            Dimensionality of the feedforward network in the encoder.
        - dropout: float
            Dropout rate for regularization.
        """
        super(MALDITransformer, self).__init__()
        self.d_model = d_model                  # Store the embedding dimension

        # Input projection layer to map intensity and mz values to embedding space
        self.input_proj = nn.Linear(2, d_model)  # Projects [mz, intensity] to a vector of size d_model

        # Positional encoding module to add positional information to embeddings
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        # Define a Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,                    # Embedding dimension
            nhead=nhead,                        # Number of attention heads
            dim_feedforward=dim_feedforward,    # Dimension of the feedforward network
            dropout=dropout                     # Dropout rate
        )

        # Stack multiple Transformer encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers               # Number of encoder layers to stack
        )

        # Classification head to map the encoder outputs to class logits
        self.classifier = nn.Linear(d_model, num_classes)  # Output layer for classification

    def forward(self, mz, intensity, padding_mask=None):
        """
        Forward pass of the model.

        Parameters:
        - mz: torch.Tensor
            Tensor of m/z values with shape (batch_size, seq_length).
        - intensity: torch.Tensor
            Tensor of intensity values with shape (batch_size, seq_length).

        Returns:
        - logits: torch.Tensor
            Output logits for each class with shape (batch_size, num_classes).
        """
        # Normalize inputs
        # mz = mz / 20000.0  # Assuming max m/z is 20000
        # intensity = intensity / intensity.max(dim=1, keepdim=True)[0]
        
        # Reshape intensity and mz to add a feature dimension (batch_size, seq_length, 1)
        intensity = intensity.unsqueeze(-1)
        mz = mz.unsqueeze(-1)
        
        # Concatenate mz and intensity to create input features
        x = torch.cat([mz, intensity], dim=-1)  # Shape: (batch_size, seq_length, 2)
        
        # Project input features into the embedding space
        x = self.input_proj(x)  # Shape: (batch_size, seq_length, d_model)
        
        # Transpose dimensions to match Transformer input requirements
        x = x.transpose(0, 1)   # Shape: (seq_length, batch_size, d_model)
        mz = mz.transpose(0, 1).squeeze(-1)  # Shape: (seq_length, batch_size)
        
        # Apply positional encoding using mz values
        x = self.positional_encoding(x, mz)
        
        # Pass the embeddings through the Transformer encoder
        x = self.transformer_encoder(x)  # Shape: (seq_length, batch_size, d_model)
        
        # Aggregate the encoder outputs across the sequence dimension
        x = x.mean(dim=0)  # Shape: (batch_size, d_model)
        
        # Pass the aggregated output through the classification head to get logits
        logits = self.classifier(x)  # Shape: (batch_size, num_classes)
        
        return logits

class PositionalEncoding(nn.Module):
    
    # Positional encoding module to add positional information to embeddings based on m/z values.
    
    def __init__(self, d_model, dropout=0.1):
        
    # Initialize the PositionalEncoding module.

        """ Parameters:
        - d_model: int
            Dimensionality of the embeddings.
        - dropout: float
            Dropout rate for regularization. """ 
        
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)  # Dropout layer for regularization
        self.d_model = d_model                # Embedding dimension

    def forward(self, x, positions):
        
        # Forward pass of the PositionalEncoding module.

        """ Parameters:
        - x: torch.Tensor
            Input tensor with shape (seq_length, batch_size, d_model).
        - positions: torch.Tensor
            Tensor of m/z values with shape (seq_length, batch_size).

        Returns:
        - x: torch.Tensor
            Tensor with positional encodings added. """
        
        # Normalize positions to the range [0, 1]
        positions_norm = positions / positions.max()

        # Expand positions to match x's last dimension
        positions_norm = positions_norm.unsqueeze(-1)  # Shape: (seq_length, batch_size, 1)

        # Compute the positional encodings
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=x.device) *
            (-np.log(10000.0) / self.d_model)
        )

        pe = torch.zeros_like(x, device=x.device)  # Initialize positional encoding tensor

        # Compute sinusoidal functions of positions
        pe[:, :, 0::2] = torch.sin(positions_norm * div_term)
        pe[:, :, 1::2] = torch.cos(positions_norm * div_term)

        # Add positional encodings to the input embeddings
        x = x + pe

        # Apply dropout
        return self.dropout(x)