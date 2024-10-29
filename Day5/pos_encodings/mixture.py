import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class MixturePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000, dropout=0.1):
        """
        Initialize the Mixture Positional Encoding module.

        This function sets up a combination of absolute and relative positional encodings:
        1. Store the model dimension and maximum sequence length.
        2. Create a dropout layer for regularization.
        3. Generate absolute positional encoding:
           a. Create a position tensor and a division term tensor.
           b. Apply sine to even indices and cosine to odd indices of the encoding.
           c. Register the encoding as a buffer (non-trainable parameter).
        4. Create an embedding layer for relative positional encoding.
        5. Initialize a learnable parameter alpha for mixing absolute and relative encodings.

        The mixture of absolute and relative encodings allows the model to capture both
        global position information and local relative position information.
        """
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(p=dropout)

        # Absolute position encoding
        pe = torch.zeros(max_seq_len, d_model)  # Step 1: Initialize position encoding tensor.
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)  # Step 2: Create position tensor.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))  # Step 3: Compute division term.
        pe[:, 0::2] = torch.sin(position * div_term)  # Step 4: Apply sine to even indices.
        pe[:, 1::2] = torch.cos(position * div_term)  # Step 5: Apply cosine to odd indices.
        pe = pe.unsqueeze(0)  # Step 6: Add batch dimension.
        self.register_buffer('pe', pe)  # Step 7: Register the encoding as a non-trainable buffer.

        # Relative position encoding
        self.relative_position_embedding = nn.Embedding(2 * max_seq_len - 1, d_model)  # Step 8: Create relative position embedding.

        # Mixture parameter
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Step 9: Initialize the mixture parameter alpha.

    def forward(self, x):
        """
        Apply the mixture of absolute and relative positional encodings to the input.

        This function combines absolute and relative positional encodings:
        1. Extract the sequence length from the input.
        2. Get the absolute positional encoding for the current sequence length.
        3. Generate relative position indices.
        4. Get the relative positional encoding from the embedding layer.
        5. Mix absolute and relative encodings using the learnable alpha parameter.
        6. Apply dropout to the mixed encoding and return.

        This process allows the model to leverage both absolute and relative
        position information, with the ability to learn the optimal mixture.
        """
        seq_len = x.size(1)  # Step 1: Extract sequence length from input.
        
        # Absolute position encoding
        absolute_pe = self.pe[:, :seq_len]  # Step 2: Retrieve absolute positional encoding.

        # Relative position encoding
        relative_pos = torch.arange(-seq_len + 1, seq_len, device=x.device).unsqueeze(0)  # Step 3: Generate relative position indices.
        relative_pe = self.relative_position_embedding(relative_pos + self.max_seq_len - 1)  # Step 4: Get relative positional encoding.

        # Mix absolute and relative encodings
        mixed_pe = self.alpha * absolute_pe + (1 - self.alpha) * relative_pe[:, :seq_len]  # Step 5: Combine encodings using alpha.

        return self.dropout(mixed_pe)  # Step 6: Apply dropout and return mixed encoding.

class MixtureAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, max_seq_len=5000):
        """
        Initialize the Mixture Attention module.

        This function sets up the attention mechanism with mixture positional encoding:
        1. Store the model dimension, number of heads, and maximum sequence length.
        2. Create the mixture positional encoding module.
        3. Create projection layers for queries, keys, values, and output.
        4. Create a dropout layer for regularization.

        This setup allows the attention mechanism to use a mixture of absolute and
        relative positional information in its calculations.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        self.pos_encoding = MixturePositionalEncoding(d_model, max_seq_len, dropout)  # Step 1: Initialize positional encoding module.
        
        self.q_proj = nn.Linear(d_model, d_model)  # Step 2: Create linear projection for queries.
        self.k_proj = nn.Linear(d_model, d_model)  # Step 3: Create linear projection for keys.
        self.v_proj = nn.Linear(d_model, d_model)  # Step 4: Create linear projection for values.
        self.out_proj = nn.Linear(d_model, d_model)  # Step 5: Create linear projection for output.
        
        self.dropout = nn.Dropout(dropout)  # Step 6: Initialize dropout layer.

    def forward(self, x):
        """
        Perform the attention mechanism with mixture positional encoding.

        This function applies the entire attention process:
        1. Extract batch size and sequence length from input.
        2. Generate positional encoding using the mixture method.
        3. Add positional encoding to the input.
        4. Project input to queries, keys, and values:
           a. Apply linear transformations (q_proj, k_proj, v_proj).
           b. Reshape the result to separate the heads.
           c. Transpose dimensions to put the head dimension second.
        5. Compute attention scores:
           a. Multiply queries and keys (matrix multiplication).
           b. Scale the result by sqrt(d_model).
        6. Apply softmax and dropout to get attention weights.
        7. Apply attention weights to values.
        8. Reshape and project the result back to the original dimension.
        9. Return the final output.

        This process allows the model to attend to different parts of the input
        sequence, taking into account both absolute and relative positional information.
        """
        batch_size, seq_len, _ = x.shape  # Step 1: Extract batch size and sequence length.
        pos_enc = self.pos_encoding(x)  # Step 2: Generate positional encoding.

        # Add positional encoding to input
        x = x + pos_enc  # Step 3: Add positional encoding to input.

        # Project input to queries, keys, and values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)  # Step 4a: Apply linear transformation for queries.
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)  # Step 4b: Apply linear transformation for keys.
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)  # Step 4c: Apply linear transformation for values.

        attn = (q @ k.transpose(-2, -1)) / (self.d_model ** 0.5)  # Step 5a: Compute attention scores by multiplying queries and keys.
        attn = self.dropout(attn.softmax(dim=-1))  # Step 6: Apply softmax and dropout to get attention weights.

        out = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, self.d_model)  # Step 7: Apply attention weights to values and reshape.
        return self.out_proj(out)  # Step 8: Project output back to original dimension.

    def visualize_positional_encoding(self, seq_len=100):
        """
        Visualize the mixture positional encoding.

        This function creates two types of visualizations:
        1. Heatmap of the positional encoding:
           a. Generate the positional encoding for a zero input tensor.
           b. Create a heatmap using seaborn.
           c. Set title (including the current alpha value), x-label, and y-label.
           d. Display the plot.
        2. Line plot of encoding components:
           a. For every 10th dimension, plot its values across positions.
           b. Set title (including the current alpha value), x-label, y-label, and legend.
           c. Display the plot.

        These visualizations help in understanding the patterns and characteristics
        of the mixture positional encoding, showing how absolute and relative
        information are combined.
        """
        pos_enc = self.pos_encoding(torch.zeros(1, seq_len, self.d_model)).squeeze().detach().cpu().numpy()  # Step 1: Generate positional encoding for visualization.

        # Heatmap visualization
        plt.figure(figsize=(12, 6))
        sns.heatmap(pos_enc, cmap='coolwarm')  # Step 2: Create heatmap.
        plt.title(f"Mixture Positional Encoding (α={self.pos_encoding.alpha.item():.2f})")  # Step 3: Set title.
        plt.xlabel("Encoding Dimension")  # Step 4: Set x-label.
        plt.ylabel("Position")  # Step 5: Set y-label.
        plt.show()  # Step 6: Display the plot.

        # Line plot visualization
        plt.figure(figsize=(12, 6))
        for i in range(0, pos_enc.shape[1], pos_enc.shape[1] // 10):  # Step 1: Iterate through dimensions.
            plt.plot(pos_enc[:, i], label=f'Dim {i}')  # Step 2: Plot each dimension.
        plt.title(f"Mixture Positional Encoding Components (α={self.pos_encoding.alpha.item():.2f})")  # Step 3: Set title.
        plt.xlabel("Position")  # Step 4: Set x-label.
        plt.ylabel("Encoding Value")  # Step 5: Set y-label.
        plt.legend()  # Step 6: Show legend.
        plt.show()  # Step 7: Display the plot.

if __name__ == "__main__":
    d_model = 512
    num_heads = 8
    max_seq_len = 1000
    mixture_attn = MixtureAttention(d_model, num_heads, max_seq_len=max_seq_len)

    x = torch.randn(32, 100, d_model)  # Step 1: Create random input tensor.
    output = mixture_attn(x)  # Step 2: Apply the Mixture Attention mechanism.

    # Step 3: Print shapes and mixture parameter
    print(f"Input shape: {x.shape}")  
    print(f"Output shape: {output.shape}")  
    print(f"Mixture parameter α: {mixture_attn.pos_encoding.alpha.item():.4f}")  

    mixture_attn.visualize_positional_encoding()  # Step 4: Visualize positional encoding.