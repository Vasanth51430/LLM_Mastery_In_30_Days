import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Initialize the MultiHeadAttention module.

        This function sets up the multi-head attention mechanism:
        1. Storing the model dimension, number of attention heads, and calculating the head dimension.
        2. Creating linear projection layers for queries, keys, values, and the final output.
        3. Setting up a dropout layer for regularization.

        Multi-head attention allows the model to jointly attend to information from different representation subspaces.
        """
        super().__init__()
        self.d_model = d_model  # Step 1a: Model dimension
        self.num_heads = num_heads  # Step 1b: Number of attention heads
        self.head_dim = d_model // num_heads  # Step 1c: Dimension of each attention head
        
        # Step 2: Linear projections for queries, keys, values, and output
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)  # Step 3: Dropout layer

    def forward(self, q, k, v, mask=None):
        """
        Perform the forward pass of multi-head attention.

        This function applies the attention mechanism:
        1. Project the input queries, keys, and values using the corresponding linear layers.
        2. Reshape the projected tensors to separate the heads.
        3. Compute attention scores using scaled dot-product.
        4. Apply the mask if provided, to prevent attention to certain positions.
        5. Calculate attention probabilities and apply dropout.
        6. Multiply attention probabilities by values to get the output.
        7. Project the output back to the original dimension.

        Args:
            q (Tensor): Queries of shape (batch_size, seq_len, d_model).
            k (Tensor): Keys of shape (batch_size, seq_len, d_model).
            v (Tensor): Values of shape (batch_size, seq_len, d_model).
            mask (Tensor, optional): Mask of shape (batch_size, 1, 1, seq_len) to prevent attention to certain positions.

        Returns:
            Tensor: The output of shape (batch_size, seq_len, d_model).
        """
        batch_size = q.size(0)  # Step 1: Get batch size

        # Step 2: Project input to queries, keys, and values
        q = self.q_proj(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # Shape: (batch_size, num_heads, seq_len, head_dim)
        k = self.k_proj(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # Shape: (batch_size, num_heads, seq_len, head_dim)
        v = self.v_proj(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # Shape: (batch_size, num_heads, seq_len, head_dim)

        # Step 3: Calculate attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # Shape: (batch_size, num_heads, seq_len, seq_len)

        # Step 4: Apply mask (if provided)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))  # Prevent attention to masked positions

        # Step 5: Calculate attention probabilities
        attn_probs = self.dropout(torch.softmax(attn_scores, dim=-1))  # Shape: (batch_size, num_heads, seq_len, seq_len)

        # Step 6: Apply attention to values
        out = torch.matmul(attn_probs, v).transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # Shape: (batch_size, seq_len, d_model)

        return self.out_proj(out)  # Step 7: Project output back to original dimension

    def visualize_attention(self, attn_weights, head_idx=0):
        """
        Visualize the attention weights for a specific head.

        This function creates a heatmap to show the attention weights:
        1. Use seaborn to create a heatmap of the attention weights for the specified head.
        2. Set titles and labels for better understanding.

        Args:
            attn_weights (Tensor): Attention weights of shape (batch_size, num_heads, seq_len, seq_len).
            head_idx (int): Index of the attention head to visualize (default is 0).
        """
        fig, ax = plt.subplots(figsize=(10, 8))  # Step 1: Create a new figure for the heatmap
        sns.heatmap(attn_weights[0, head_idx].detach().cpu().numpy(), ax=ax, cmap='viridis')  # Step 2: Create heatmap
        ax.set_title(f"Multi-Head Attention Weights for Head {head_idx + 1}")  # Set title
        ax.set_xlabel("Key Position")  # Set x-label
        ax.set_ylabel("Query Position")  # Set y-label
        plt.show()  # Display heatmap

if __name__ == "__main__":
    d_model = 512  # Define the model dimension
    num_heads = 8  # Define the number of attention heads
    mha = MultiHeadAttention(d_model, num_heads)  # Instantiate the MultiHeadAttention module

    seq_len = 100  # Define the sequence length
    x = torch.randn(32, seq_len, d_model)  # Create a random input tensor of shape (batch_size, seq_len, d_model)
    output = mha(x, x, x)  # Apply the multi-head attention mechanism

    # Print shapes
    print(f"Input shape: {x.shape}")  # Display input shape
    print(f"Output shape: {output.shape}")  # Display output shape

    with torch.no_grad():
        # Compute attention weights for visualization
        q = mha.q_proj(x).view(32, seq_len, num_heads, -1).transpose(1, 2)  # Shape: (32, num_heads, seq_len, head_dim)
        k = mha.k_proj(x).view(32, seq_len, num_heads, -1).transpose(1, 2)  # Shape: (32, num_heads, seq_len, head_dim)
        attn_weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / (mha.head_dim ** 0.5), dim=-1)  # Shape: (32, num_heads, seq_len, seq_len)

    mha.visualize_attention(attn_weights)  # Visualize the attention weights
