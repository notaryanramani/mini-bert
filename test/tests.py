from modelkit import BERT
from datakit import WikipediaScraper
from datakit import DataLoader
import unittest
import torch

class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.vocab_size = 1000
        self.n_embd = 64
        self.block_size = 32
        self.n_heads = 4
        self.n_layers = 4
        self.dropout = 0.2

        self.model = BERT(self.vocab_size, self.n_embd, self.block_size,
                          self.n_heads, self.n_layers, self.dropout)
        
        self.data_loader = DataLoader(filepath='data/data.txt', block_size=self.block_size, batch_size=16)
        
    
    def test_logits_shape(self):
        x = torch.randint(0, self.vocab_size, (16, self.block_size))
        logits, _ = self.model(x)
        self.assertEqual(logits.shape, (16, self.vocab_size))

    def test_batch_size(self):
        x, y = self.data_loader.get_batch()
        self.assertEqual(x.shape, (16, self.block_size))
        self.assertEqual(y.shape, (16,))

if __name__ == '__main__':
    unittest.main()
