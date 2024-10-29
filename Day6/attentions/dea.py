import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

class DisentangledAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len, dropout=0.1):
        super().__init__()
        # Ensure model dimension is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize model parameters
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  # Dimension of each head
        self.max_seq_len = max_seq_len  # Maximum sequence length
        
        # Define linear projection layers for queries, keys, and values
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)  # Final output projection
        
        # Initialize positional embeddings
        self.pos_embed = nn.Parameter(torch.Tensor(max_seq_len, self.head_dim))
        nn.init.xavier_uniform_(self.pos_embed)  # Xavier initialization
        
        # Initialize learnable parameters for attention score adjustments
        self.alpha = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.beta = nn.Parameter(torch.zeros(num_heads, 1, 1))
        
        self.dropout = nn.Dropout(dropout)  # Dropout for regularization

    def forward(self, x, mask=None):
        """
        Step 1: Get batch size and sequence length from input tensor x.
        Step 2: Project input tensor x to queries (q), keys (k), and values (v).
        Step 3: Compute content-based attention scores.
        Step 4: Compute position-based attention scores using relative positions.
        Step 5: Combine content and position attention scores.
        Step 6: Apply the mask if provided.
        Step 7: Calculate attention probabilities using softmax.
        Step 8: Compute output by applying attention probabilities to values.
        Step 9: Apply final output projection.
        """
        batch_size, seq_len, _ = x.size()  # Step 1: Get batch size and sequence length
        
        # Step 2: Project queries, keys, and values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Step 3: Compute content-based attention scores
        content_attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Step 4: Compute position-based attention scores
        pos_attn = self.get_relative_positions(seq_len)
        
        # Step 5: Combine content and position attention scores
        attn_scores = content_attn + (self.alpha * pos_attn + self.beta)
        
        # Step 6: Apply mask to attention scores if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        
        # Step 7: Calculate attention probabilities
        attn_probs = self.dropout(F.softmax(attn_scores, dim=-1))
        
        # Step 8: Compute output by applying attention probabilities to values
        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Step 9: Apply final output projection
        return self.out_proj(out)
    
    def get_relative_positions(self, seq_len):
        """
        Step 1: Slice the positional embeddings to match the sequence length.
        Step 2: Compute the relative position scores by matrix multiplication.
        Step 3: Return the position-based attention weights with an added dimension.
        """
        pos_embed = self.pos_embed[:seq_len]  # Step 1: Slice positional embeddings
        return torch.matmul(pos_embed, pos_embed.transpose(0, 1)).unsqueeze(0)  # Step 2: Compute relative positions
    
    def visualize_attention(self, attn_weights, head_idx=0):
        """
        Step 1: Create a new figure for visualizing attention weights.
        Step 2: Plot total attention weights as a heatmap for the specified head.
        Step 3: Calculate and plot position-based attention weights as a heatmap.
        Step 4: Adjust layout and display the plots.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Step 2: Plot total attention weights as a heatmap
        sns.heatmap(attn_weights[0, head_idx].detach().cpu().numpy(), ax=ax1, cmap='viridis')
        ax1.set_title(f"Total Attention Weights for Head {head_idx + 1}")
        ax1.set_xlabel("Key Position")
        ax1.set_ylabel("Query Position")
        
        # Step 3: Plot position-based attention weights
        seq_len = attn_weights.size(-1)
        pos_attn = self.get_relative_positions(seq_len).squeeze(0).detach().cpu().numpy()
        sns.heatmap(pos_attn, ax=ax2, cmap='viridis')
        ax2.set_title(f"Position-based Attention Weights")
        ax2.set_xlabel("Key Position")
        ax2.set_ylabel("Query Position")
        
        plt.tight_layout()  # Step 4: Adjust layout
        plt.show()

if __name__ == "__main__":
    d_model = 512
    num_heads = 8
    max_seq_len = 100
    disentangled_attn = DisentangledAttention(d_model, num_heads, max_seq_len)

    seq_len = 50
    x = torch.randn(32, seq_len, d_model)  # Create random input tensor
    output = disentangled_attn(x)  # Apply Disentangled Attention

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    with torch.no_grad():
        # Get attention weights for visualization
        q = disentangled_attn.q_proj(x).view(32, seq_len, disentangled_attn.num_heads, disentangled_attn.head_dim).transpose(1, 2)
        k = disentangled_attn.k_proj(x).view(32, seq_len, disentangled_attn.num_heads, disentangled_attn.head_dim).transpose(1, 2)
        
        # Compute attention scores for content and position
        content_attn = torch.matmul(q, k.transpose(-2, -1)) / (disentangled_attn.head_dim ** 0.5)
        pos_attn = disentangled_attn.get_relative_positions(seq_len)
        
        # Combine attention scores
        attn_scores = content_attn + (disentangled_attn.alpha * pos_attn + disentangled_attn.beta)
        attn_weights = F.softmax(attn_scores, dim=-1)  # Apply softmax to get attention weights

    disentangled_attn.visualize_attention(attn_weights)  # Visualize attention weights