from modelkit.architecture import Encoder, PositionEmbedding
import torch.nn as nn
import torch.nn.functional as F


dropout = 0.2

class BERT:
    def __init__(self, vocab_size, n_embd, block_size, n_heads, n_layers, dropout = dropout):
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = PositionEmbedding(n_embd, block_size)
        self.encoders = nn.Sequential(*[Encoder(n_embd, n_heads) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(2 * n_embd)
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(2 * n_embd, vocab_size)


    def forward(self, x, targets = None):
        x = self.tok_emb(x)
        x = self.pos_emb(x)

        x = self.encoders(x)
        x = self.ln(x)
        logits = self.linear(x)

        if targets is not None:
            loss = F.cross_entropy(logits, loss)
        else:
            loss = None

        return logits, loss
    

    def predict(self, x):
        logits, _ = self(x)
        return logits