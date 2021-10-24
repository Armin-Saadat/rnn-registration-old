class Trainer:
    def __init__(self, model, dataloader, loss, optimizer):
        self.model = model.train()
        self.dataloader = dataloader
        self.loss = loss
        self.optimizer = optimizer

    def train(self):

