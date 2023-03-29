class NoamOpt:
    """
        Implementation of the custom optimiser used in Attention Is All You Need.
        The equation that varies the learning rate is:
            lr = d_model ^ -0.5 * min(step ^ -0.5, step * warmup ^ -0.5)

        It corresponds to increasing the lr linearly for the first warmup steps then 
        decreasing it proportionally to the inverse square root of the step number.
        In the paper, they used warmup as 4000
    """
    
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))