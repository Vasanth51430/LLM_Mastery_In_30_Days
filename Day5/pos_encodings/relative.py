import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import math

class RelativeGlobalAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_len=1024, dropout=0.1):
        """
        Initializes the RelativeGlobalAttention module.

        - `d_model`: Dimensionality of the model (input embeddings).
        - `num_heads`: Number of attention heads.
        - `max_len`: Maximum sequence length supported (for positional embeddings).
        - `dropout`: Dropout rate for regularization.

        Steps:
        1. Compute the dimensionality of each attention head (`d_head`).
        2. Define linear transformations for query, key, and value projections.
        3. Initialize the relative positional embedding matrix (`Er`).
        4. Create a lower triangular mask for causal attention.
        """
        super().__init__()

        # Split the model dimension by the number of heads to get head-specific dimension
        d_head, remainder = divmod(d_model, num_heads)
        if remainder:
            raise ValueError("incompatible `d_model` and `num_heads`")  # Ensure `d_model` is divisible by `num_heads`
        
        # Store maximum sequence length, model dimensionality, and number of attention heads
        self.max_len = max_len
        self.d_model = d_model
        self.num_heads = num_heads

        # Linear projections for key, value, and query vectors
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout)

        # Learnable relative positional embeddings of shape (max_len, d_head)
        self.Er = nn.Parameter(torch.randn(max_len, d_head))

        # Create a lower triangular mask for causal attention (future positions cannot be attended to)
        self.register_buffer("mask", torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(0))
        # Shape of mask: (1, 1, max_len, max_len)

    def forward(self, x):
        """
        Performs the forward pass for attention.

        Steps:
        1. Reshape the input sequence into key, value, and query projections.
        2. Compute the relative positional embedding bias.
        3. Calculate attention scores by combining standard query-key dot product with the relative positional bias.
        4. Apply causal masking to ensure future tokens are not attended to.
        5. Normalize the attention scores using softmax.
        6. Multiply the attention weights with value vectors to get the final output.

        Parameters:
        - `x`: Input tensor of shape (batch_size, seq_len, d_model)
        """
        # Get the shape of input sequence
        batch_size, seq_len, _ = x.shape
        
        # Check if the sequence length exceeds the allowed maximum length
        if seq_len > self.max_len:
            raise ValueError("sequence length exceeds model capacity")

        # 1. Compute the key, value, and query projections from the input
        # Each projection has shape (batch_size, seq_len, num_heads, d_head)
        k_t = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        # 2. Calculate the relative positional embeddings for the current sequence length
        # We select embeddings for positions from (max_len - seq_len) to max_len
        start = self.max_len - seq_len
        Er_t = self.Er[start:, :].transpose(0, 1)  # Transpose to shape (d_head, seq_len)

        # 3. Compute the bias term using query and relative positional embeddings
        # This bias is added to attention scores later to encode relative position info
        QEr = torch.matmul(q, Er_t)  # Shape: (batch_size, num_heads, seq_len, seq_len)
        
        # Apply the 'skew' operation to shift the bias matrix for relative positions
        Srel = self.skew(QEr)

        # 4. Calculate the standard attention scores using the dot product of queries and keys
        # Attention scores: (batch_size, num_heads, seq_len, seq_len)
        QK_t = torch.matmul(q, k_t)
        
        # 5. Combine standard attention scores with the relative positional bias
        # Divide by square root of head dimension to normalize scores
        attn = (QK_t + Srel) / math.sqrt(q.size(-1))

        # 6. Apply the causal mask to prevent attending to future positions
        mask = self.mask[:, :, :seq_len, :seq_len]
        attn = attn.masked_fill(mask == 0, float("-inf"))

        # 7. Apply softmax to get the attention weights and dropout for regularization
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # Multiply attention weights by value vectors
        
        # 8. Reshape the output back to original sequence format
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        
        return self.dropout(out), attn  # Return attention for visualization
    
    def skew(self, QEr):
        """
        Performs the skew operation for relative positional embedding matrix.
        
        Steps:
        1. Pad the input matrix QEr on the left.
        2. Reshape the padded matrix to shift rows by one position.
        3. Remove the padded column and return the skewed matrix.

        Parameters:
        - `QEr`: Tensor of shape (batch_size, num_heads, seq_len, seq_len) representing query-relative positional embeddings interaction.
        """
        # 1. Add padding to the left of the input matrix (shift columns to the right)
        padded = F.pad(QEr, (1, 0))
        
        # 2. Reshape the padded matrix to shift rows by one position
        batch_size, num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)

        # 3. Remove the padded column and return the skewed matrix (relative positional bias)
        Srel = reshaped[:, :, 1:, :]
        return Srel
    
    def visualize_attention(self, attn, seq_len):
        """
        Visualizes the attention matrix for a given sequence.

        Steps:
        1. Select the attention weights for the first batch and first head.
        2. Generate a heatmap to visualize attention scores across query and key positions.

        Parameters:
        - `attn`: Tensor of shape (batch_size, num_heads, seq_len, seq_len) representing the attention weights.
        - `seq_len`: Length of the sequence.
        """
        # 1. Extract attention weights for the first head and first batch
        attn_weights = attn[0, 0, :, :].detach().cpu().numpy()

        # 2. Plot heatmap using seaborn to visualize attention distribution
        plt.figure(figsize=(10, 8))
        sns.heatmap(attn_weights, cmap='coolwarm', xticklabels=range(seq_len), yticklabels=range(seq_len))
        plt.title('Relative Position Embeddings Attention Weights Visualization')
        plt.xlabel('Key Positions')
        plt.ylabel('Query Positions')
        plt.show()

if __name__ == "__main__":
    # Sample input parameters
    seq_len = 10  # Sequence length
    d_model = 64  # Model dimensionality
    num_heads = 4  # Number of attention heads

    # Initialize the RelativeGlobalAttention model
    model = RelativeGlobalAttention(d_model=d_model, num_heads=num_heads, max_len=1024)

    # Generate a random input tensor with shape (batch_size=1, seq_len, d_model)
    x = torch.randn(1, seq_len, d_model)

    # Perform the forward pass
    out, attn = model(x)

    # Visualize the attention matrix
    model.visualize_attention(attn, seq_len)
