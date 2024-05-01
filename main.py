'''
This can be a sample code for training the model.
This code trains a model on the text file located at data/data.txt and saves the model to the model folder.
'''

from modelkit import AutoModel
import torch
from datakit import DataLoaderForBase
from trainer import Trainer
from torch.optim import AdamW
import os

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    filepath = 'data/data.txt'

    d = DataLoaderForBase(filepath)
    d.set_device(device)
    tok = d.get_tokenizer()

    m = AutoModel().load(vocab_size=tok.n_vocab)
    m.to(device)
    optim = AdamW(m.parameters(), lr=1e-4)

    t = Trainer(m, optim, d)
    m = t.train(epochs=1, steps_per_epoch=10)
    m.to(device)

    # models_dir = 'model'
    # os.makedirs(models_dir, exist_ok=True)

    # torch.save(m.state_dict(), 'model/model.pth')
    