import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

class ALiBiAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Initialize the ALiBiAttention module.

        This function sets up the ALiBi attention mechanism, including:
        1. Storing the model dimension, number of attention heads, and calculating the head dimension.
        2. Creating linear projection layers for queries, keys, values, and output.
        3. Setting up a dropout layer for regularization.
        4. Calculating ALiBi slopes to create the bias for each attention head.

        The ALiBi mechanism allows the model to incorporate relative position information in attention scores,
        enhancing its ability to capture the relationships between tokens.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Linear projections for queries, keys, values, and output
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # Calculate ALiBi slopes
        m = torch.arange(1, self.num_heads + 1)
        m = 1.0 / (2 ** (8 * m / self.num_heads))  # Step 4: Calculate slopes for each head
        self.register_buffer("m", m.view(1, self.num_heads, 1, 1))  # Step 5: Register slopes as a buffer

    def forward(self, x):
        """
        Compute the forward pass of the ALiBi attention mechanism.

        This function applies the attention mechanism with ALiBi bias:
        1. Extract the batch size and sequence length from the input tensor.
        2. Project the input to queries, keys, and values using the corresponding linear layers.
        3. Calculate the attention scores using scaled dot-product.
        4. Add the ALiBi bias based on the position differences between tokens.
        5. Apply softmax to the attention scores and dropout to obtain attention weights.
        6. Multiply the attention weights by the values to get the output.
        7. Project the output back to the original dimension.

        The use of ALiBi bias allows the model to incorporate relative positional information,
        improving its performance on tasks requiring attention to token relationships.
        """
        batch_size, seq_len, _ = x.shape  # Step 1: Extract batch size and sequence length.

        # Project input to queries, keys, and values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # Step 2a: Project queries
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # Step 2b: Project keys
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # Step 2c: Project values

        # Calculate attention scores
        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # Step 3: Compute scaled dot-product attention scores

        # Add ALiBi bias
        alibi_bias = -torch.abs(torch.arange(seq_len, device=x.device).unsqueeze(0) - torch.arange(seq_len, device=x.device).unsqueeze(1))  # Step 4a: Compute position difference
        alibi_bias = self.m * alibi_bias.unsqueeze(0).unsqueeze(0)  # Step 4b: Scale by the slopes
        attn_scores = attn_scores + alibi_bias  # Step 4c: Add ALiBi bias to attention scores

        attn_weights = self.dropout(attn_scores.softmax(dim=-1))  # Step 5: Apply softmax and dropout to get attention weights

        out = (attn_weights @ v).transpose(1, 2).reshape(batch_size, seq_len, self.d_model)  # Step 6: Multiply by values and reshape
        return self.out_proj(out)  # Step 7: Project output back to original dimension

    def visualize_alibi_bias(self, seq_len=100):
        """
        Visualize the ALiBi bias for different attention heads.

        This function creates two types of visualizations:
        1. Heatmap of the ALiBi bias for each attention head:
           a. Calculate the bias for the specified sequence length.
           b. Create a heatmap for each head showing bias values.
        2. Line plot of bias values for a specific query position across all heads:
           a. Plot the bias values for the middle query position.

        These visualizations help in understanding how ALiBi biases vary across attention heads
        and their effects on the attention mechanism.
        """
        alibi_bias = -torch.abs(torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1))  # Step 1: Compute bias matrix
        alibi_bias = alibi_bias.unsqueeze(0)  # Step 2: Reshape to (1, seq_len, seq_len)

        m_expanded = self.m.squeeze().unsqueeze(1).unsqueeze(2)  # Step 3: Expand slopes to match shape for broadcasting
        alibi_bias = m_expanded * alibi_bias  # Step 4: Broadcast and compute final ALiBi bias

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # Step 5: Create subplots for heatmaps
        fig.suptitle("ALiBi Bias for Different Attention Heads")

        for i, ax in enumerate(axes.flat):
            if i < self.num_heads:
                sns.heatmap(alibi_bias[i].detach().cpu().numpy(), ax=ax, cmap='coolwarm')  # Step 6: Plot heatmaps
                ax.set_title(f"Head {i + 1}")  # Set title for each subplot
                ax.set_xlabel("Query Position")  # Set x-label
                ax.set_ylabel("Key Position")  # Set y-label
            else:
                ax.axis('off')  # Hide unused subplots

        plt.tight_layout()  # Adjust layout
        plt.show()  # Display heatmaps

        plt.figure(figsize=(12, 6))  # Step 7: Create line plot for middle query position
        for i in range(self.num_heads):
            plt.plot(alibi_bias[i, seq_len // 2], label=f'Head {i + 1}')  # Plot bias for middle query position
        plt.title("ALiBi Bias for Different Heads (Middle Query)")  # Set title
        plt.xlabel("Key Position")  # Set x-label
        plt.ylabel("Bias Value")  # Set y-label
        plt.legend()  # Show legend
        plt.show()  # Display line plot

if __name__ == "__main__":
    d_model = 512
    num_heads = 8
    alibi_attn = ALiBiAttention(d_model, num_heads)

    x = torch.randn(32, 100, d_model)  # Step 1: Create random input tensor.
    output = alibi_attn(x)  # Step 2: Apply the ALiBi attention mechanism.

    # Step 3: Print shapes
    print(f"Input shape: {x.shape}")  
    print(f"Output shape: {output.shape}")  

    alibi_attn.visualize_alibi_bias()  # Step 4: Visualize ALiBi bias.
