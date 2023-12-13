from typing import Optional, Tuple, Union
from torch import Tensor
from torch.optim import Adam
from torch.optim.optimizer import params_t

class Adam_Torch(Adam):
    def __init__(self, name:str, params: params_t=None, learning_rate: float = 0.001) -> None:
        if params is not None:
            super().__init__(params, learning_rate)
            # super().__init__(params, learning_rate, betas, eps, weight_decay, amsgrad, maximize=maximize, capturable=capturable, differentiable=differentiable)
        else:
            super().__init__([{'params': []}], learning_rate)
            # super().__init__([{'params': []}], learning_rate, betas, eps, weight_decay, amsgrad, maximize=maximize, capturable=capturable, differentiable=differentiable)
            # super().__init__(params=[{'params': []}],lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, maximize=maximize, capturable=capturable, differentiable=differentiable)
        self.name = name

if __name__ == '__main__':
    pass