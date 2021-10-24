class Trainer:
    def __init__(self, args, model, dataloader, optimizer, sim_loss, smooth_loss, seg_loss):
        self.args = args
        self.model = model.train()
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.sim_loss = sim_loss
        self.smooth_loss = smooth_loss
        self.seg_loss = seg_loss

    def train(self):
        pass
