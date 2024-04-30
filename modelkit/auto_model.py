from .model import BERT, BERTforClassification
import os
import torch
import tiktoken
import warnings
import gdown

class AutoModel():
    def __init__(self) -> None:
        pass

    def load(self, vocab_size = None, n_targets = None, task = 'base', load_weights:bool = False):
        if vocab_size is None:
            warnings.warn('Vocab size not given, taking default vocab size. This may lead to error if not using default tokenzier.')
            vocab_size = 50260 # Default vocab size of mini-bert tokenizer
        
        if task == 'cls' and n_targets is None:
            raise Exception("Classification model needs no. of output labels. Specify 'n_targets' parameters")

        if load_weights:
            url = 'https://drive.google.com/uc?id=14YsUh8UfwrRt4cB95JH26sDQNPtt7hZl'
            path = os.path.join(os.getcwd(), 'models', 'model.pth')
            if not os.path.exists(path):
                print('File not found.')
                os.makedirs('models', exist_ok=True)
                gdown.download(url, path, quiet=False, fuzzy=True)
            sd = torch.load(path)

        if task == 'base':
            m = BERT(vocab_size = vocab_size)
            if load_weights:
                m.load_state_dict(sd)
            return m
        
        elif task == 'cls':
            m = BERTforClassification(vocab_size = vocab_size, n_targets = n_targets)
            if load_weights:
                state_dict = {key:value for key, value in sd.keys() if not key.startswith('linear')}
                m.load_state_dict(state_dict, strict=False)
            return m
                

