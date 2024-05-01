from tqdm import tqdm
from utils import CustomLogger
import os

class Trainer():
    def __init__(
            self, 
            model, 
            optimizer, 
            data_loader,
            ):
        self.m = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        os.makedirs('logs', exist_ok=True)
        self.logger = CustomLogger(__name__, 'logs/train_logs.log')


    def train(self, epochs = 5, steps_per_epoch = 4000):
        print('Training Started')
        for epoch in range(epochs):
            s = tqdm(range(steps_per_epoch), leave=False, ncols=100)
            for _ in s:
                x, y = self.data_loader.get_batch()
                _, loss = self.m(x, targets = y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                s.set_description(f'Epoch: {epoch}/{epochs}')
                s.set_postfix(loss = loss.item())
            train_loss, val_loss = self.__eval()
            self.logger.info(f'Epoch {epoch} - Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}')
        print('Training Finished. Check logs for accuracy & loss values')
        return self.m
    
    def __eval(self):
        val_lossi = []
        train_lossi = []
        for _ in range(10):
            x, y = self.data_loader.get_batch()
            _, loss = self.m(x, y)
            train_lossi.append(loss.item())

            x, y = self.data_loader.get_batch('val')
            _, loss = self.m(x, y)
            val_lossi.append(loss.item())
        
        train_average_loss = sum(train_lossi) / len(train_lossi)
        val_average_loss = sum(val_lossi) / len(val_lossi)

        return train_average_loss, val_average_loss


