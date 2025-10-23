import math
from typing import Union

import torch
from torch.optim.lr_scheduler import LambdaLR
from torchtune.training.memory import OptimizerInBackwardWrapper

def get_wsd(
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: Union[int, float],
        num_training_steps: int,
        drop_iter: Union[int, float] = 0.8, 
        last_epoch = -1,
    ) -> LambdaLR:

    if 0 <= num_warmup_steps < 1:
        num_warmup_steps = int(num_warmup_steps * num_training_steps)
        print(f"Given num_warmup_steps {num_warmup_steps} is a fraction of num_training_steps {num_training_steps}, converting to int: {num_warmup_steps}")

    if 0 <= drop_iter < 1:
        drop_iter = int(drop_iter * num_training_steps)
        print(f"Given drop_iter {drop_iter} is a fraction of num_training_steps {num_training_steps}, converting to int: {drop_iter}")
    
    def lr_lambda(current_step: int) -> float:
        # linear warmup phase
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)

        # decay phase
        if current_step > num_training_steps - drop_iter:
            progress = (num_training_steps - current_step) / drop_iter
            return (0.1 + max(0.9 * progress, 0)) 
            
        # stable phase
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch)