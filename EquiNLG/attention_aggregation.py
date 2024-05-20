import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionAggregation(nn.Module):
    def __init__(self, group_size, vocab_size, seq_len, device='cuda'):
        super().__init__()
        self.group_size = group_size
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.dim = 128  # Dimension for the attention mechanism
        self.device = device

        # Linear layers to compute attention scores
        self.query = nn.Linear(vocab_size, self.dim)
        self.key = nn.Linear(vocab_size, self.dim)
        self.value = nn.Linear(vocab_size, self.dim)

        # Linear layer to project back to vocab_size
        self.output_projection = nn.Linear(self.dim, vocab_size)

    def forward(self, group_logits):
        """
        Computes attention weights and applies them to the logits.

        Args:
            group_logits: Tensor of shape [group_size, batch_size, seq_len, vocab_size]

        Returns:
            Aggregated logits with attention weights applied.
        """
        group_size, batch_size, seq_len, vocab_size = group_logits.shape
        
        # Reshape group_logits to [group_size, batch_size * seq_len, vocab_size]
        group_logits_reshaped = group_logits.view(group_size, batch_size * seq_len, vocab_size)

        # Compute queries, keys, and values
        queries = self.query(group_logits_reshaped)  # [group_size, batch_size * seq_len, dim]
        keys = self.key(group_logits_reshaped)  # [group_size, batch_size * seq_len, dim]
        values = self.value(group_logits_reshaped)  # [group_size, batch_size * seq_len, dim]

        # Compute attention scores
        attention_scores = torch.einsum('gij,gik->gij', queries, keys) / torch.sqrt(torch.tensor(self.dim, dtype=torch.float32).to(self.device))  # [group_size, batch_size * seq_len]
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=0)  # [group_size, batch_size * seq_len]

        # Compute weighted sum of logits
        weighted_logits = torch.einsum('gij,gik->ik', attention_weights, values)  # [batch_size * seq_len, dim]

        # Reshape back to [batch_size, seq_len, vocab_size]
        #weighted_logits = weighted_logits.view(batch_size, seq_len, vocab_size)
        
        #weighted_logits = weighted_logits.view(batch_size, seq_len, self.dim)  # Corrected reshaping based on actual size
        #weighted_logits = self.value.inverse(weighted_logits).view(batch_size, seq_len, vocab_size)  # Corrected reshaping based on actual size
        weighted_logits = self.output_projection(weighted_logits).view(batch_size, seq_len, vocab_size)  # [batch_size, seq_len, vocab_size]

        return weighted_logits
