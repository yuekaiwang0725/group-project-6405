import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================
# Bahdanau Attention Component
# Logic: Computes a context vector by taking a weighted sum of all hidden 
# states, where weights are determined by the relevance of each token.
# =========================================================================
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        """
        Initialize learnable weight matrices.
        Equation: score = v^T * tanh(W * H)
        """
        # Learnable weight matrix W for projecting hidden states
        self.w_omega = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        # Learnable vector v for scoring the importance of each token
        self.u_omega = nn.Parameter(torch.Tensor(hidden_size, 1))
        
        # Initialize parameters using uniform distribution for stability
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def forward(self, H, pad_mask=None):
        """
        Args:
            H: Input hidden states from GRU (Batch_Size, Seq_Len, Feature_Dim)
            pad_mask: Binary mask for <PAD> tokens (Batch_Size, Seq_Len, 1)
        """
        # 1. Non-linear transformation of hidden states
        # Operation: u = tanh(H * W)
        u = torch.tanh(torch.matmul(H, self.w_omega))
        
        # 2. Calculate raw attention scores
        # Operation: att = u * v
        att = torch.matmul(u, self.u_omega)
        
        # 3. Apply Padding Mask
        # If mask exists, set <PAD> scores to -infinity to force weights to zero
        if pad_mask is not None:
            att = att.masked_fill(pad_mask, -1e9)

        # 4. Normalize scores via Softmax to get Attention Weights (alpha)
        # alpha represents the percentage contribution of each token
        alpha = F.softmax(att, dim=1)
        
        # 5. Compute Context Vector (V) via weighted sum
        # Operation: V = Σ (alpha_i * H_i)
        scored_x = H * alpha
        V = torch.sum(scored_x, dim=1)
        
        # Return context vector for classification and alpha for visualization
        return V, alpha


# =========================================================================
# Main Model: Bi-GRU + Attention Architecture
# Logic Flow: Embedding -> Bi-GRU -> Attention -> FC -> Classification
# =========================================================================
class GRUAttention(nn.Module):
    def __init__(self, config):
        super(GRUAttention, self).__init__()
        
        # [Layer 1] Embedding Layer
        # Maps discrete token IDs to dense continuous vectors.
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        
        # [Layer 2] Bi-directional GRU (The "Engine" of Context)
        # ---------------------------------------------------------------------
        # 💡 KEY EXPLANATION FOR PRESENTATION:
        # 1. Dual Processing: By setting bidirectional=True, PyTorch creates 
        #    TWO separate GRU layers: one processing left-to-right (Forward) 
        #    and one right-to-left (Backward).
        # 2. Complete Context: This allows each word to "know" its past 
        #    and "foresee" its future, creating a richer word representation.
        # ---------------------------------------------------------------------
        self.gru = nn.GRU(
            config.embedding_dim, 
            config.hidden_dim, 
            bidirectional=True,  # This activates the dual-direction logic
            batch_first=True
        )
        
        # [Layer 3] Bahdanau Attention
        # 💡 Note: Input dimension is config.hidden_dim * 2 because the 
        # Bi-GRU output concatenates the Forward and Backward states.
        self.attention = BahdanauAttention(hidden_size=config.hidden_dim * 2)
        
        # [Layer 4] Output Layer (Classifier)
        # Maps the final context vector to the number of target classes.
        self.fc = nn.Linear(config.hidden_dim * 2, config.num_classes)

    def forward(self, x):
        """
        Main Forward Pass (Defines the data flow through the network)
        x shape: (Batch_Size, Seq_Len)
        """
        # Create mask for padding tokens to ensure the model ignores empty slots
        pad_mask = (x == 0).unsqueeze(-1)

        # Step 1: Embedding Lookup
        # Result: (Batch, Seq_Len, Embed_Dim)
        emb = self.embedding(x)
        
        # Step 2: Temporal Feature Extraction (Bi-GRU Logic)
        # ---------------------------------------------------------------------
        # 💡 WHY ONLY ONE LINE FOR BI-DIRECTIONAL?
        # Even though we only call 'self.gru' once, the internal C++ engine 
        # processes the sequence in both directions simultaneously. 
        #
        # H Output Shape: (Batch, Seq_Len, Hidden_Dim * 2)
        # The last dimension is doubled because PyTorch joins the 128-dim 
        # Forward state with the 128-dim Backward state into one 256-dim vector.
        # ---------------------------------------------------------------------
        H, _ = self.gru(emb)
        
        # Step 3: Global Feature Refinement via Attention
        # V: Context Vector (Batch, Hidden_Dim * 2) - The "Sum" of important words
        # alpha: Weights (Batch, Seq_Len, 1) - Shows which words the model focused on
        V, alpha = self.attention(H, pad_mask)
        
        # Step 4: Final Classification
        # Result: (Batch, Num_Classes)
        out = self.fc(V)
        
        return out, alpha