import tiktoken
import torch

class DataLoader:
    def __init__(self, filepath:str, block_size:int, batch_size:int, tokenizer_name:str = 'r50k_base'):
        self.filepath = filepath
        self.tokenizer_name = tokenizer_name
        self.block_size = block_size
        self.batch_size = batch_size
        self.text = open(self.filepath, 'r').read()
        base = tiktoken.get_encoding(self.tokenizer_name)
        self.mask_token = base.n_vocab
        self.tokenizer = tiktoken.Encoding(
            name = 'mini-bert-tokenizer',
            pat_str = base._pat_str,
            mergeable_ranks = base._mergeable_ranks,
            special_tokens = {
                ** base._special_tokens,
                '<|mask|>' : self.mask_token
            }
        )
        self.tokens = self.tokenizer.encode(self.text)
    
    def get_text(self):
        return self.text
    
    def get_data(self):
        start_indeces = torch.randint(0, len(self.tokens) - self.block_size, (self.batch_size,))
        data = [self.tokens[i.item(): i.item()+self.block_size] for i in start_indeces]
        data = torch.tensor(data)
        x, y = self.get_batch(data)
        return x, y

    def get_batch(self, data):
        x, y = [], []

        change_probs = torch.tensor([0.8, 0.1, 0.1])
        for d in data:
            d = d.clone()
            idx = torch.randint(0, data.shape[1], (1,))[0].item()
            yi = d[idx].item()
            change_flag = torch.multinomial(input = change_probs, num_samples = 1).item()
            if change_flag == 0:
                d[idx] = self.mask_token
            elif change_flag == 1:
                d[idx] = torch.randint(0, self.tokenizer.n_vocab - 1, (1,))[0].item()

            x.append(d.tolist())
            y.append(yi)

        x = torch.tensor(x)
        y = torch.tensor(y)
        return x, y
    
    def __repr__(self):
        repr_text = f'''Filepath - {self.filepath} | Tokenizer - {self.tokenizer_name}'''
        return repr_text