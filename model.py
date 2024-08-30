import torch
import torch.nn as nn
import numpy as np

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = torch.softmax(scores, dim=-1)
        return torch.matmul(attention, value), attention

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.layernorm1(x + self.dropout(attn_output))
        ff_output = self.feedforward(x)
        x = self.layernorm2(x + self.dropout(ff_output))
        return x

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, num_classes_sentiment, num_classes_classification):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self._generate_positional_encoding(d_model, max_length=500)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.sentiment_head = nn.Linear(d_model, num_classes_sentiment)
        self.classification_head = nn.Linear(d_model, num_classes_classification)

    def forward(self, x, task="sentiment"):
        x = self.embedding(x) + self.positional_encoding[:x.size(1), :].to(x.device)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # Use the output corresponding to the [CLS] token
        cls_output = x[:, 0, :]

        if task == "sentiment":
            return self.sentiment_head(cls_output)
        elif task == "classification":
            return self.classification_head(cls_output)

    def _generate_positional_encoding(self, d_model, max_length):
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
