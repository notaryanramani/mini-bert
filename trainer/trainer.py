from tqdm import tqdm
from utils import CustomLogger
import os
import torch
from torch.optim import AdamW

class Trainer():
    def __init__(self, model, data_loader, lr, logfile = 'logs/train_logs.log', enable_checkpointing = False, checkpoint_dir = 'checkpoints'):
        self.m = model
        self.optimizer = AdamW(self.m.parameters(), lr=lr)
        self.data_loader = data_loader

        os.makedirs('logs', exist_ok=True)
        self.logger = CustomLogger(__name__, logfile)

        self.enable_checkpointing = enable_checkpointing
        if enable_checkpointing:
            self.checkpoint_dir = checkpoint_dir
            os.makedirs(checkpoint_dir, exist_ok=True)


    def train(self, epochs = 5, steps_per_epoch = 4000, eval_size = 50):
        print('Training Started')
        for epoch in range(epochs):
            lp = tqdm(range(steps_per_epoch), leave=False, ncols=100)
            for j in lp:
                x, y = self.data_loader.get_batch()
                _, loss = self.m(x, targets = y)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                lp.set_description(f'Epoch [{epoch+1}/{epochs}]')
                lp.set_postfix(loss = loss.item())
            train_loss, val_loss, train_acc, val_acc = self.__eval(eval_size = eval_size)
            self.logger.info(f'Epoch {epoch+1} - Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}, Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}')
            if self.enable_checkpointing:
                torch.save(self.m.state_dict(), f'{self.checkpoint_dir}/checkpoint_{epoch+1}_{train_loss=}_{train_acc=}.pth')
        print('Training Finished. Check logs for accuracy & loss values')
        return self.m
    
    def __eval(self, eval_size):
        '''
            evaluates the model on both training and validation data
        '''

        val_lossi = []
        train_lossi = []
        train_acc = []
        val_acc = []
        for _ in range(eval_size):
            x, y = self.data_loader.get_batch()
            logits, loss = self.m(x, y)
            y_hat = torch.argmax(logits, dim=1)
            train_accuracy = self.__get_accuracy(y, y_hat)
            train_lossi.append(loss.item())
            train_acc.append(train_accuracy)


            x, y = self.data_loader.get_batch('val')
            logits, loss = self.m(x, y)
            y_hat = torch.argmax(logits, dim=1)
            val_accuracy = self.__get_accuracy(y, y_hat)
            val_lossi.append(loss.item())
            val_acc.append(val_accuracy)
        
        train_average_loss = sum(train_lossi) / len(train_lossi)
        val_average_loss = sum(val_lossi) / len(val_lossi)

        train_average_accuracy = sum(train_acc) / len(train_acc)
        val_average_accuracy = sum(val_acc) / len(val_acc)

        return train_average_loss, val_average_loss, train_average_accuracy, val_average_accuracy
    
    def __get_accuracy(self, y, y_hat):
        '''
            calculates the accuracy of the model
        '''
        correct = []
        for yi, y_hati in zip(y, y_hat):
            if yi == y_hati:
                correct.append(1)
            else:
                correct.append(0)
        return sum(correct) / len(correct)

