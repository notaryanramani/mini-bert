class Trainer():
    def __init__(
            self, 
            model, 
            optimizer, 
            data_loader,
            steps = 20000):
        
        self.m = model
        self.optimizer = optimizer
        self.steps = steps
        self.step_eval = self.steps // 10
        self.data_loader = data_loader


    def train(self):
        print('Training Started')
        for step in range(self.steps):
            x, y = self.data_loader.get_batch()
            _, loss = self.m(x, targets = y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step % self.step_eval == 0:
                train_loss, val_loss = self.__eval()
                print(f'Train Loss: {train_loss}, Val Loss: {val_loss}')

        return self.m
    
    def __eval(self):
        val_lossi = []
        train_lossi = []
        for _ in range(50):
            x, y = self.data_loader.get_batch()
            _, loss = self.m(x, y)
            train_lossi.append(loss.item())

            x, y = self.data_loader.get_batch('val')
            _, loss = self.m(x, y)
            val_lossi.append(loss.item())
        
        train_average_loss = sum(train_lossi) / len(train_lossi)
        val_average_loss = sum(val_lossi) / len(val_lossi)

        return train_average_loss, val_average_loss


