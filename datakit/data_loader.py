import tiktoken
import torch
import pandas as pd
from typing import List

class DataLoaderForBase:

    def __init__(self, filepath:str, block_size:int =  128, batch_size:int = 32, tokenizer_name:str = 'r50k_base'):
        self.filepath = filepath
        self.tokenizer_name = tokenizer_name
        self.block_size = block_size
        self.batch_size = batch_size
        with open(self.filepath, 'r', encoding='utf-8') as f:
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
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.train_val_split()
        
        
    def train_val_split(self):
        tokens = self.tokenizer.encode(self.text, allowed_special=self.tokenizer.special_tokens_set)
        split = int(len(tokens) * 0.9)
        self.train_tokens = tokens[:split]
        self.val_tokens = tokens[split:]
    

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

        '''
        Prepares the batch for training.

        Parameters:
            - data : tensor of shape (batch_size, block_size)

        Returns:
            - x : tensor of shape (batch_size, block_size)
            - y : tensor of shape (batch_size,)
        '''

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

        x = torch.tensor(x).to(self.device)
        y = torch.tensor(y).to(self.device)
        return x, y
    

    def get_data(self):
        return self.text
    

    def get_tokenizer(self):
        return self.tokenizer
    

    def set_device(self, device:str):
        assert device in ['cuda', 'cpu'], 'Device should be either cuda or cpu'
        self.device = device

    
class DataLoaderForClassification(DataLoaderForBase):

    def __init__(self, x:List[str], y:List[int], block_size:int =  128, batch_size:int = 32, tokenizer_name:str = 'r50k_base'):
        assert all([isinstance(xi, str) for xi in x]), 'All elements of x should be of type str'
        assert all([isinstance(yi, int) for yi in y]), 'All elements of y should be of type int'

        self.x = x
        self.y = y
        self.block_size = block_size
        self.batch_size = batch_size
        self.tokenizer_name = tokenizer_name

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
        self.train_val_split()
    

    def train_val_split(self, split_ratio:float = 0.9):
        split = int(len(self.x) * split_ratio)
        self.train_x = self.x[:split]
        self.train_y = self.y[:split]
        self.val_x = self.x[split:]
        self.val_y = self.y[split:]


    def get_batch(self, split='train'):
        '''
        Returns the batch for training.

        Parameters:
            - split: type of split i.e., train or validation

        Returns:
            - x : tensor of shape (batch_size, block_size)
            - y : tensor of shape (batch_size,)
        '''
        x = self.train_x if split == 'train' else self.val_x
        y = self.train_y if split == 'train' else self.val_y
        start_indeces = torch.randint(0, len(x), (self.batch_size,))
        x = [x[i] for i in start_indeces]
        y = [y[i] for i in start_indeces]
        x = self.prepare_batch(x)
        x = torch.tensor(x)
        y = torch.tensor(y)
        return x, y
    

    def prepare_batch(self, x):

        '''
        Prepares the batch for training.

        Parameters:
            - x : list of strings

        Returns:
            - x_ : list of lists of integers that are the tokens of the input strings
        '''

        x_ = []

        for xi in x:
            xi_ = self.tokenizer.encode(xi, allowed_special=self.tokenizer.special_tokens_set)
            if len(xi_) < self.block_size:
                xi_ += [self.tokenizer.special_tokens['<|pad|>']] * (self.block_size - len(xi))
            elif len(xi_) > self.block_size:
                xi_ = xi_[:self.block_size]
            x_.append(xi_)
        
        return x_


    def get_data(self):
        return self.x, self.y
    

