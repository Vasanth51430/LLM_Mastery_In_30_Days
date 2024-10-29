import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_groups, dropout=0.1):
        super().__init__()
        # Ensure that the number of heads can be evenly divided into groups
        assert num_heads % num_groups == 0, "num_heads must be divisible by num_groups"
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.heads_per_group = num_heads // num_groups  # Heads per group
        self.head_dim = d_model // num_heads  # Dimension per head
        
        # Define linear projection layers for queries and key-value pairs
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)  # Combined projection for keys and values
        self.out_proj = nn.Linear(d_model, d_model)  # Final output projection
        
        self.dropout = nn.Dropout(dropout)  # Dropout for regularization

    def forward(self, q, kv, mask=None):
        """
        Step 1: Get batch size and sequence length from the query tensor.
        Step 2: Project the query tensor (q) using the query projection layer (q_proj).
        Step 3: Project the key and value tensor (kv) using the combined key-value projection layer (kv_proj).
        Step 4: Split the output of the kv projection into keys (k) and values (v).
        Step 5: Reshape the queries, keys, and values for grouping.
        Step 6: Compute attention scores using the queries and keys.
        Step 7: Apply the mask (if provided) to the attention scores.
        Step 8: Calculate attention probabilities using softmax on the attention scores.
        Step 9: Compute the output by multiplying attention probabilities with values.
        Step 10: Reshape the output back to the original dimension and apply the output projection.
        """
        batch_size = q.size(0)  # Get batch size
        seq_len = q.size(1)  # Get sequence length
        
        # Step 2: Project queries
        q = self.q_proj(q).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Step 3: Project key-value pairs
        k, v = self.kv_proj(kv).chunk(2, dim=-1)  # Split into keys and values
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Step 5: Group queries, keys, and values
        q = q.view(batch_size, self.num_groups, self.heads_per_group, seq_len, self.head_dim)
        k = k.view(batch_size, self.num_groups, self.heads_per_group, seq_len, self.head_dim)
        v = v.view(batch_size, self.num_groups, self.heads_per_group, seq_len, self.head_dim)
        
        # Step 6: Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Step 7: Apply mask to attention scores (if provided)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Step 8: Calculate attention probabilities
        attn_probs = self.dropout(torch.softmax(attn_scores, dim=-1))
        
        # Step 9: Compute output by applying attention to values
        out = torch.matmul(attn_probs, v)
        # Step 10: Reshape output and apply final projection
        out = out.transpose(2, 3).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(out)

    def visualize_attention(self, attn_weights, group_idx=0, head_idx=0):
        """
        Step 1: Create a new figure for the heatmap.
        Step 2: Use seaborn to create a heatmap of the attention weights for the specified group and head.
        Step 3: Set the title of the heatmap.
        Step 4: Set labels for the x-axis and y-axis.
        Step 5: Display the heatmap.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        # Step 2: Plot attention weights as a heatmap
        sns.heatmap(attn_weights[0, group_idx, head_idx].detach().cpu().numpy(), ax=ax, cmap='viridis')
        ax.set_title(f"Grouped Query Attention Weights for Group {group_idx + 1}, Head {head_idx + 1}")
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")
        plt.show()

if __name__ == "__main__":
    d_model = 512
    num_heads = 8
    num_groups = 4
    gqa = GroupedQueryAttention(d_model, num_heads, num_groups)

    seq_len = 100
    x = torch.randn(32, seq_len, d_model)  # Create random input tensor
    output = gqa(x, x)  # Apply Grouped Query Attention

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    with torch.no_grad():
        # Get attention weights for visualization
        q = gqa.q_proj(x).view(32, seq_len, gqa.num_heads, gqa.head_dim).transpose(1, 2)
        k, _ = gqa.kv_proj(x).chunk(2, dim=-1)  # Get keys
        k = k.view(32, seq_len, gqa.num_heads, gqa.head_dim).transpose(1, 2)
        
        # Reshape queries and keys for visualization
        q = q.view(32, gqa.num_groups, gqa.heads_per_group, seq_len, gqa.head_dim)
        k = k.view(32, gqa.num_groups, gqa.heads_per_group, seq_len, gqa.head_dim)
        
        # Calculate attention weights
        attn_weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / (gqa.head_dim ** 0.5), dim=-1)

    gqa.visualize_attention(attn_weights)  # Visualize attention weights
