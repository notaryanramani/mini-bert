import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionEmbedding(nn.Module):

    ''' Position Embeddings for the BERT architecture'''

    def __init__(self, n_embd, block_size) :
        super().__init__()
        pe = torch.zeros(block_size, n_embd)
        pos = torch.arange(0, block_size).float().unsqueeze(1)
        div_term = torch.tensor([10000.0]) ** (torch.arange(0, n_embd, 2) / n_embd)
        pe[:, 0::2] = torch.sin(pos / div_term)
        pe[:, 1::2] = torch.sin(pos / div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        out = self.pe + x
        return out


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

    ''' Multi Head Attention '''

    def __init__(self, n_embd, n_heads, head_size, dropout = 0.2):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([h(x, x, x) for h in self.heads], dim=-1)
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

    '''Encoder for BERT, with both Forward and Backward attentions'''

    def __init__(self, n_embd, n_heads, dropout = 0.2) -> None:
        super().__init__()
        assert n_embd % n_heads == 0, 'n_head is not divisible by n_embd'
        head_size = n_embd // n_heads
        self.forward_heads = MultiHeadAttention(n_embd, n_heads, head_size, dropout=dropout)
        self.backward_heads = MultiHeadAttention(n_embd, n_heads, head_size, dropout=dropout)
        self.forward_ffn = FeedForward(2 * n_embd, dropout=dropout)
        self.f_ln = nn.LayerNorm(n_embd)
        self.b_ln = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        for_x = self.ln1(x)
        for_x = self.forward_heads(for_x) + for_x
        back_x = torch.flip(x, dims=(-1, ))
        back_x = self.b_ln(back_x)
        back_x = self.backward_heads(back_x) + back_x
        x = torch.cat([for_x, back_x], dim=-1)
        x = self.ln2(x)
        x = self.ffn(x) + x
        out = self.drop(x)
        return out

