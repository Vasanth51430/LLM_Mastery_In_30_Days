import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class MultiMethodAbsolutePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000, dropout=0.1, method='sinusoidal'):
        """
        Initialize the MultiMethodAbsolutePositionalEncoding module.

        This function sets up various positional encoding methods, including:
        1. Storing the model dimension and maximum sequence length.
        2. Creating a dropout layer for regularization.
        3. Initializing positional encodings based on the selected method (sinusoidal, learned, algebraic, or binary).

        Positional encoding allows the model to incorporate information about the position of tokens in the input sequence.
        """
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.max_seq_len = max_seq_len
        self.method = method

        # Initialize positional encoding based on the selected method
        if method == 'sinusoidal':
            self.pe = self._sinusoidal_encoding()
        elif method == 'learned':
            self.pe = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        elif method == 'algebraic':
            self.pe = self._algebraic_encoding()
        elif method == 'binary':
            self.pe = self._binary_encoding()
        else:
            raise ValueError(f"Unknown encoding method: {method}")

    def _sinusoidal_encoding(self):
        """
        Compute sinusoidal positional encoding.

        This function creates a positional encoding using sine and cosine functions:
        1. Initialize a zero tensor of shape (max_seq_len, d_model).
        2. Create a tensor of positions from 0 to max_seq_len.
        3. Compute division terms using exponential and logarithmic functions.
        4. Fill even indices of the encoding with sine of (position * div_term).
        5. Fill odd indices of the encoding with cosine of (position * div_term).
        6. Return the encoding tensor with an extra dimension at the start.

        The sinusoidal encoding allows the model to easily learn to attend to relative positions.
        """
        pe = torch.zeros(self.max_seq_len, self.d_model)  # Step 1
        position = torch.arange(0, self.max_seq_len, dtype=torch.float).unsqueeze(1)  # Step 2
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))  # Step 3
        pe[:, 0::2] = torch.sin(position * div_term)  # Step 4
        pe[:, 1::2] = torch.cos(position * div_term)  # Step 5
        return pe.unsqueeze(0)  # Step 6

    def _algebraic_encoding(self):
        """
        Compute algebraic positional encoding.

        This function creates a positional encoding using a power series:
        1. Initialize a zero tensor of shape (max_seq_len, d_model).
        2. Create a tensor of positions from 0 to max_seq_len.
        3. For each dimension i, compute (pos / (max_seq_len - 1)) ^ (i / (d_model - 1)).
        4. Return the encoding tensor with an extra dimension at the start.

        This encoding provides a unique pattern for each position.
        """
        pe = torch.zeros(self.max_seq_len, self.d_model)  # Step 1
        position = torch.arange(0, self.max_seq_len, dtype=torch.float).unsqueeze(1)  # Step 2
        for i in range(self.d_model):  # Step 3
            pe[:, i] = position.squeeze() / (self.max_seq_len - 1) ** (i / (self.d_model - 1))
        return pe.unsqueeze(0)  # Step 4

    def _binary_encoding(self):
        """
        Compute binary positional encoding.

        This function creates a positional encoding using binary representation:
        1. Initialize a zero tensor of shape (max_seq_len, d_model).
        2. For each position from 0 to max_seq_len:
           a. Convert the position to binary.
           b. Fill the encoding tensor with binary digits (0 or 1).
        3. Return the encoding tensor with an extra dimension at the start.

        This encoding represents each position as its binary number.
        """
        pe = torch.zeros(self.max_seq_len, self.d_model)  # Step 1
        for pos in range(self.max_seq_len):  # Step 2
            for i in range(min(self.d_model, len(bin(self.max_seq_len)) - 2)):  # Step 2a
                pe[pos, i] = (pos >> i) & 1  # Step 2b
        return pe.unsqueeze(0)  # Step 3

    def forward(self, x):
        """
        Add positional encoding to the input tensor.

        This function applies the positional encoding to the input:
        1. Take a slice of the pre-computed positional encoding (pe) to match the input sequence length.
        2. Add this positional encoding to the input tensor.
        3. Apply dropout to the result.
        4. Return the output.

        This process allows the model to incorporate position information into the input representation.
        """
        x = x + self.pe[:, :x.size(1)]  # Step 1 & 2
        return self.dropout(x)  # Step 3 & 4

class AbsolutePositionAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, max_seq_len=5000, method='sinusoidal'):
        """
        Initialize the AbsolutePositionAttention module.

        This function sets up the attention mechanism with positional encoding:
        1. Storing the model dimension, number of attention heads, and creating linear projection layers for queries, keys, values, and output.
        2. Initializing the MultiMethodAbsolutePositionalEncoding for positional encoding.
        3. Setting up a dropout layer for regularization.

        This attention mechanism allows the model to incorporate positional information while attending to different parts of the input.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.pos_encoding = MultiMethodAbsolutePositionalEncoding(d_model, max_seq_len, dropout, method)  # Step 2

        # Linear projections for queries, keys, values, and output
        self.q_proj = nn.Linear(d_model, d_model)  # Step 1a
        self.k_proj = nn.Linear(d_model, d_model)  # Step 1b
        self.v_proj = nn.Linear(d_model, d_model)  # Step 1c
        self.out_proj = nn.Linear(d_model, d_model)  # Step 1d
        
        self.dropout = nn.Dropout(dropout)  # Step 3

    def forward(self, x):
        """
        Perform the attention mechanism with positional encoding.

        This function applies the entire attention process:
        1. Extract batch size and sequence length from input.
        2. Apply positional encoding to the input.
        3. Project input to queries, keys, and values:
           a. Apply linear transformations (q_proj, k_proj, v_proj).
           b. Reshape the result to separate the heads.
           c. Transpose dimensions to put the head dimension second.
        4. Compute attention scores:
           a. Multiply queries and keys (matrix multiplication).
           b. Scale the result by sqrt(d_model).
           c. Apply softmax to get attention weights.
           d. Apply dropout to the attention weights.
        5. Apply attention to values:
           a. Multiply attention weights with values.
           b. Transpose and reshape the result to combine heads.
        6. Project the result back to the original dimension (out_proj).
        7. Return the final output.

        This process allows the model to attend to different parts of the input sequence, taking into account positional information.
        """
        batch_size, seq_len, _ = x.shape  # Step 1

        x = self.pos_encoding(x)  # Step 2
        
        # Project input to queries, keys, and values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)  # Step 3a
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)  # Step 3b
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)  # Step 3c
        
        # Calculate attention scores
        attn = (q @ k.transpose(-2, -1)) / (self.d_model ** 0.5)  # Step 4a
        attn = self.dropout(attn.softmax(dim=-1))  # Step 4b & 4c

        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, self.d_model)  # Step 5a & 5b
        return self.out_proj(out)  # Step 6 & 7
    
    def visualize_positional_encoding(self, seq_len=100):
        """
        Visualize the positional encoding used in the model.

        This function creates two visualizations of the positional encoding:
        1. Heatmap:
           a. Extract the positional encoding tensor
           b. Create a heatmap using seaborn
           c. Set title, x-label, and y-label
           d. Display the plot
        2. Line plot:
           a. Create a new figure
           b. For every 10th dimension, plot its values across positions
           c. Set title, x-label, y-label, and legend
           d. Display the plot

        These visualizations help in understanding the patterns and 
        characteristics of the chosen positional encoding method.
        """
        pos_enc = self.pos_encoding.pe[:, :seq_len, :].squeeze().detach().cpu().numpy()
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(pos_enc, cmap='coolwarm')
        plt.title(f"Absolute Positional Encoding ({self.pos_encoding.method})")
        plt.xlabel("Encoding Dimension")
        plt.ylabel("Position")
        plt.show()
        
        plt.figure(figsize=(12, 6))
        for i in range(0, pos_enc.shape[1], pos_enc.shape[1]//10):
            plt.plot(pos_enc[:, i], label=f'Dim {i}')
        plt.title(f"Absolute Positional Encoding Components ({self.pos_encoding.method})")
        plt.xlabel("Position")
        plt.ylabel("Encoding Value")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    # Test the implementation with different encoding methods
    d_model = 512
    num_heads = 8
    max_seq_len = 1000

    methods = ['sinusoidal', 'learned', 'algebraic', 'binary']

    for method in methods:
        print(f"\nTesting {method.capitalize()} Encoding:")
        abs_pos_attn = AbsolutePositionAttention(d_model, num_heads, max_seq_len=max_seq_len, method=method)

        x = torch.randn(32, 100, d_model)  # Sample input tensor
        output = abs_pos_attn(x)

        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")

        abs_pos_attn.visualize_positional_encoding()