class evaluator:
    def __init__(self, exp_id, model, dataloader, loss):
        self.exp_id = exp_id
        self.model = model
        self.dataloader = dataloader
        self.loss = loss


