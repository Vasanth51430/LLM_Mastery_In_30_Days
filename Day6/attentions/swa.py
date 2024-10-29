import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

class SlidingWindowAttention(nn.Module):
    def __init__(self, d_model, num_heads, window_size, dropout=0.1):
        super().__init__()
        # Initialize model parameters
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  # Dimension of each head
        self.window_size = window_size  # Size of the sliding window
        
        # Define linear projection layers for queries, keys, and values
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)  # Final output projection
        
        self.dropout = nn.Dropout(dropout)  # Dropout for regularization

    def forward(self, x, mask=None):
        """
        Step 1: Get batch size and sequence length from the input tensor x.
        Step 2: Project the input tensor x to queries (q), keys (k), and values (v).
        Step 3: Create a sliding window mask to restrict attention.
        Step 4: If a mask is provided, combine it with the window mask.
        Step 5: Compute attention scores using queries and keys.
        Step 6: Apply the window mask to the attention scores.
        Step 7: Calculate attention probabilities using softmax on the masked scores.
        Step 8: Compute the output by applying attention probabilities to values.
        Step 9: Apply the final output projection to the attention output.
        """
        batch_size, seq_len, _ = x.shape  # Step 1: Get batch size and sequence length
        
        # Step 2: Project queries, keys, and values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Step 3: Create a sliding window mask
        window_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        window_mask = window_mask.triu(-self.window_size // 2).tril(self.window_size // 2)  # Step 3a: Create upper and lower triangular masks
        window_mask = window_mask.unsqueeze(0).unsqueeze(0)  # Step 3b: Add batch and head dimensions
        
        # Step 4: Combine with provided mask (if exists)
        if mask is not None:
            window_mask = window_mask & mask
        
        # Step 5: Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Step 6: Apply window mask to attention scores
        attn_scores = attn_scores.masked_fill(~window_mask, float('-inf'))
        
        # Step 7: Calculate attention probabilities
        attn_probs = self.dropout(torch.softmax(attn_scores, dim=-1))
        
        # Step 8: Compute output by applying attention to values
        out = torch.matmul(attn_probs, v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Step 9: Apply final output projection
        return self.out_proj(out)

    def visualize_attention(self, attn_weights, head_idx=0):
        """
        Step 1: Create a new figure for the heatmap.
        Step 2: Use seaborn to create a heatmap of the attention weights for the specified head.
        Step 3: Set the title of the heatmap.
        Step 4: Set labels for the x-axis and y-axis.
        Step 5: Display the heatmap.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        # Step 2: Plot attention weights as a heatmap
        sns.heatmap(attn_weights[0, head_idx].detach().cpu().numpy(), ax=ax, cmap='viridis')
        ax.set_title(f"Sliding Window Attention Weights for Head {head_idx + 1}")
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")
        plt.show()

if __name__ == "__main__":
    d_model = 512
    num_heads = 8
    window_size = 16
    swa = SlidingWindowAttention(d_model, num_heads, window_size)

    seq_len = 100
    x = torch.randn(32, seq_len, d_model)  # Create random input tensor
    output = swa(x)  # Apply Sliding Window Attention

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    with torch.no_grad():
        # Get attention weights for visualization
        q = swa.q_proj(x).view(32, seq_len, swa.num_heads, swa.head_dim).transpose(1, 2)
        k = swa.k_proj(x).view(32, seq_len, swa.num_heads, swa.head_dim).transpose(1, 2)
        
        # Step 3: Create window mask for visualization
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (swa.head_dim ** 0.5)
        window_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        window_mask = window_mask.triu(-swa.window_size // 2).tril(swa.window_size // 2)
        window_mask = window_mask.unsqueeze(0).unsqueeze(0)
        
        # Apply the window mask to attention scores
        attn_scores = attn_scores.masked_fill(~window_mask, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1)

    swa.visualize_attention(attn_weights)  # Visualize attention weights