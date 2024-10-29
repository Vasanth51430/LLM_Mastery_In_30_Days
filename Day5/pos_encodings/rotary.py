import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000, base=10000):
        super().__init__()

        """
        Step-by-Step:
        1. The dimensionality of the model (d_model) is stored, which represents the size of each token embedding.
        2. Inverse frequencies are calculated using the formula `1 / (base^(2i/d_model))`. These frequencies are 
           used to scale positions based on the embedding dimensions. Frequencies only apply to even dimensions.
        3. The 'inv_freq' tensor is registered as a buffer. Buffers are persistent tensors in PyTorch that 
           are not updated during training but are saved with the model.
        4. The maximum sequence length is stored. This sets the upper limit for sequence lengths during training.
        """
        
        # Store the embedding size (d_model)
        self.d_model = d_model
        
        # Compute inverse frequencies for positional encoding
        inv_freq = 1. / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        
        # Register the inverse frequencies as a buffer to persist it
        self.register_buffer('inv_freq', inv_freq)
        
        # Store the maximum sequence length
        self.max_seq_len = max_seq_len

    def forward(self, positions):
        """
        Step-by-Step:
        1. The input 'positions' represents the index of each token in the sequence (e.g., [0, 1, 2, ...]).
        2. We compute the sinusoidal input by multiplying positions by the inverse frequencies (einsum operation).
           This generates a matrix where each position has unique sinusoidal signals across different frequencies.
        3. We generate two components: the sine and cosine values of the generated sinusoidal input.
        4. These sine and cosine values are concatenated to form the positional embeddings. The sine components 
           occupy the first half of the embedding, and the cosine components occupy the second half.
        """
        
        # Compute sinusoidal input by multiplying positions with inverse frequencies
        sinusoid_inp = torch.einsum("i,j->ij", positions.float(), self.inv_freq)
        
        # Create positional embeddings by concatenating sine and cosine components
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        
        # Return the final positional embeddings
        return emb

def rotate_half(x):
    """
    Step-by-Step:
    1. The input tensor 'x' is split into two halves along the last dimension.
    2. The second half of the tensor is negated and then concatenated with the first half.
       This forms a rotational transformation of the input.
    3. The rotation is crucial for the rotary embedding mechanism to function properly in attention layers.
    """
    
    # Split the input tensor into two halves
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    
    # Negate the second half and concatenate with the first half (rotational transformation)
    return torch.cat((-x2, x1), dim=-1)

class RotaryAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=5000):
        super().__init__()

        """
        Step-by-Step:
        1. The model dimensionality (d_model) and number of attention heads are stored. The number of heads 
           allows parallel computation of attention over smaller subspaces of the embedding.
        2. The dimension of each attention head is calculated as `d_model // num_heads`. This splits the overall 
           embedding into smaller pieces (head_dim) that each attention head can process.
        3. The 'RotaryPositionalEmbedding' is instantiated to generate the sinusoidal embeddings. These are used
           to add positional information to the query and key vectors.
        4. Linear layers are defined to project input tensors 'x' into query (Q), key (K), and value (V) tensors.
           These projections are essential in the attention mechanism.
        5. Another linear layer is created for projecting the output back to the original embedding space.
        """
        
        # Store model dimensionality and number of heads
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Calculate the dimension per head (each head processes a smaller space of the embedding)
        head_dim = d_model // num_heads
        self.head_dim = head_dim
        
        # Initialize the rotary positional embedding generator
        self.rotary_emb = RotaryPositionalEmbedding(head_dim, max_seq_len)
        
        # Linear layers to generate query (Q), key (K), and value (V) projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Linear layer for output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        """
        Step-by-Step:
        1. The input 'x' has the shape (batch_size, seq_len, d_model). We extract batch size and sequence length.
        2. 'x' is projected into query (Q), key (K), and value (V) vectors via the linear layers (q_proj, k_proj, v_proj).
           These projections are reshaped to have the shape (batch_size, num_heads, seq_len, head_dim).
        3. Positional embeddings are generated by the rotary embedding module, with 'positions' ranging from 0 to seq_len.
        4. Rotary embeddings are applied to the query and key tensors by rotating them with sine and cosine components.
        5. Attention scores are computed by taking the dot product of the modified query and key vectors and 
           normalizing them by the square root of the head dimension (scaling).
        6. Softmax is applied to normalize attention scores across the sequence length dimension.
        7. The attention scores are used to weight the value (V) vectors. The output is then projected back
           to the original embedding size via 'out_proj' and returned.
        """
        
        # Get batch size and sequence length from input shape
        batch_size, seq_len, _ = x.shape
        
        # Project input 'x' into query (Q), key (K), and value (V) vectors and reshape them
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Create a range of positions from 0 to seq_len
        positions = torch.arange(seq_len, device=x.device)
        
        # Generate rotary positional embeddings for these positions
        rot_emb = self.rotary_emb(positions)
        
        # Apply rotary embeddings to the query (Q) and key (K) vectors
        q_rot = (q * rot_emb.cos()) + (rotate_half(q) * rot_emb.sin())
        k_rot = (k * rot_emb.cos()) + (rotate_half(k) * rot_emb.sin())
        
        # Compute attention scores by taking dot product of Q and K, then scale by sqrt(head_dim)
        attn = (q_rot @ k_rot.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply softmax to normalize attention scores across the sequence length dimension
        attn = attn.softmax(dim=-1)
        
        # Use attention scores to weight the value (V) vectors, then project back to original size
        out = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        
        # Return the final output
        return self.out_proj(out)
    
    def visualize_rotary_embeddings(self, seq_len=20):
        """
        Step-by-Step:
        1. Generate a range of positions from 0 to seq_len.
        2. Compute the rotary embeddings for these positions using the 'RotaryPositionalEmbedding' module.
        3. Plot the sine and cosine components of the rotary embeddings using a heatmap.
        4. Display the visualizations, where each row represents a position and each column represents a dimension.
        """
        
        # Generate positions for rotary embeddings
        positions = torch.arange(seq_len)
        
        # Compute rotary embeddings for these positions
        rot_emb = self.rotary_emb(positions).detach().cpu().numpy()
        
        # Create a subplot to visualize the sine and cosine components
        plt.figure(figsize=(12, 6))
        
        # Plot sine component as a heatmap
        plt.subplot(1, 2, 1)
        sns.heatmap(rot_emb[:, :self.head_dim//2], cmap='coolwarm')
        plt.title("Sine Component of RoPE")
        plt.xlabel("Dimension")
        plt.ylabel("Position")
        
        # Plot cosine component as a heatmap
        plt.subplot(1, 2, 2)
        sns.heatmap(rot_emb[:, self.head_dim//2:], cmap='coolwarm')
        plt.title("Cosine Component of RoPE")
        plt.xlabel("Dimension")
        plt.ylabel("Position")
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    d_model = 512
    num_heads = 8
    max_seq_len = 1000
    rope_attn = RotaryAttention(d_model, num_heads, max_seq_len)

    x = torch.randn(32, 100, d_model)
    output = rope_attn(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    rope_attn.visualize_rotary_embeddings()
