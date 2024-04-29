import tiktoken
import torch

class DataLoader:
    def __init__(self, filepath:str, block_size:int =  32, batch_size:int = 32, tokenizer_name:str = 'r50k_base'):
        self.filepath = filepath
        self.tokenizer_name = tokenizer_name
        self.block_size = block_size
        self.batch_size = batch_size
        with open(self.filepath, 'r') as f:
            self.text = f.read()
        base = tiktoken.get_encoding(self.tokenizer_name)
        self.last_token_idx = base.n_vocab
        self.mask_token = self.last_token_idx
        self.tokenizer = tiktoken.Encoding(
            name = 'mini-bert-tokenizer',
            pat_str = base._pat_str,
            mergeable_ranks = base._mergeable_ranks,
            special_tokens = {
                ** base._special_tokens,
                '<|mask|>' : self.last_token_idx, 
                '<|sep|>' : self.last_token_idx + 1,
                '<|cls|>' : self.last_token_idx + 2
            }
        )
        tokens = self.tokenizer.encode(self.text, allowed_special=self.tokenizer.special_tokens_set)
        split = int(len(tokens) * 0.9)
        self.train_tokens = tokens[:split]
        self.val_tokens = tokens[split:]
    
    def get_text(self):
        return self.text
    
    def get_batch(self, split='train'):

        '''
        Returns the batch for training.

        Parameters:
            - split: type of split i.e., train or validation

        Returns:
            - x : tensor of shape (batch_size, block_size)
            - y : tensor of shape (batch_size,)
        '''

        tokens = self.train_tokens if split == 'train' else self.val_tokens
        start_indeces = torch.randint(0, len(tokens) - self.block_size, (self.batch_size,))
        data = [tokens[i.item(): i.item()+self.block_size] for i in start_indeces]
        data = torch.tensor(data)
        x, y = self.prepare_batch(data)
        return x, y

    def prepare_batch(self, data):
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
                d[idx] = torch.randint(0, self.tokenizer.n_vocab - len(self.tokenizer.special_tokens_set), (1,))[0].item()

            x.append(d.tolist())
            y.append(yi)

        x = torch.tensor(x)
        y = torch.tensor(y)
        return x, y
    
    def __repr__(self):
        repr_text = f'''Filepath - {self.filepath} | Tokenizer - {self.tokenizer_name}'''
        return repr_text