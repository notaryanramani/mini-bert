from modelkit.architecture import Encoder, PositionEmbedding
import torch.nn as nn
import torch.nn.functional as F


dropout = 0.2
class BERT(nn.Module):
    def __init__(self, vocab_size, n_embd = 384, block_size = 128, n_heads = 6, n_layers = 6, dropout = dropout):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = PositionEmbedding(n_embd, block_size)
        self.encoders = nn.Sequential(*[Encoder(n_embd, n_heads) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(n_embd)
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(n_embd, vocab_size)


    def forward(self, x, targets = None):
        x = self.tok_emb(x)
        x = self.pos_emb(x)

        x = self.encoders(x)
        x = self.ln(x)
        last_hidden_state = self.linear(x)
        logits = last_hidden_state.mean(dim=-2, keepdim=True).squeeze(-2)

        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
        return logits, loss
    

    def predict(self, x):
        logits, _ = self(x)
        return logits
    

    def get_parameters(self):
        params = sum([p.numel() for p in self.parameters()])
        return params
    
    def __repr__(self):
        return super().__repr__()
    

class BERTforClassification(nn.Module):
    def __init__(self, vocab_size, n_targets, n_embd = 384, block_size = 128, n_heads = 6, n_layers = 6, dropout = dropout):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = PositionEmbedding(n_embd, block_size)
        self.encoders = nn.Sequential(*[Encoder(n_embd, n_heads) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(n_embd)
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(n_embd, n_targets)


    def forward(self, x, targets = None):
        x = self.tok_emb(x)
        x = self.pos_emb(x)

        x = self.encoders(x)
        x = self.ln(x)
        last_hidden_state = self.linear(x)
        logits = last_hidden_state.mean(dim=-2, keepdim=True).squeeze(-2)

        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
        return logits, loss
    

    def predict(self, x):
        logits, _ = self(x)
        return logits
    

    def get_parameters(self):
        params = sum([p.numel() for p in self.parameters()])
        return params
    
