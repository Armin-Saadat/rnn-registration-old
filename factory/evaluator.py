from utils.losses import Dice


class Evaluator:
    def __init__(self, exp_id, model, dataloader):
        self.exp_id = exp_id
        self.model = model
        self.dataloader = dataloader
        self.dice_loss = Dice().loss

    def evaluate(self):
        pass
