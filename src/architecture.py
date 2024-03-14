import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):

    ''' A single head of attention '''

    def __init__(self, n_embd, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size)
        self.key = nn.Linear(n_embd, head_size)
        self.value = nn.Linear(n_embd, head_size)


    def forward(self, q, k, v):
        _, _, C = q.shape

        qi = self.query(q)
        ki = self.key(k)
        vi = self.value(v)

        att = qi @ ki.transpose(-1, -2 ) / torch.sqrt(torch.tensor(C, dtype=torch.float32))
        att = F.softmax(att, dim=-1)

        out = att @ vi
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_heads, head_size, dropout = 0.2):
        super().__init__()
        self.heads = nn.ModuleDict([
            Head(n_embd, head_size) for _ in range(n_heads)
        ])
        self.proj = nn.Linear(n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([h(x, x, x) for h in self.heads])
        x = self.proj(x)
        out = self.drop(x)
        return out


class FeedForward(nn.Module):

    ''' The feed forward layer in transformers '''

    def __init__(self, n_embd, dropout = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), 
            nn.Dropout(dropout)
        )

    def forward(self, x):
        out = self.net(x)
        return out


class Encoder(nn.Module):
    def __init__(self, n_embd, n_heads, dropout = 0.2) -> None:
        super().__init__()
        assert n_embd % n_heads == 0, 'n_head is not divisible by n_embd'
        head_size = n_embd // n_heads
        self.heads = MultiHeadAttention(n_embd, n_heads, head_size, dropout=dropout)
        self.ffn = FeedForward(n_embd, dropout=dropout)
        self.ln1 = nn.LayerNorm()
        self.ln2 = nn.LayerNorm()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.ln(x)
        x = self.heads(x) + x
        x = self.ln2(x)
        x = self.ffn(x) + x
        out = self.drop(x)
        return out



