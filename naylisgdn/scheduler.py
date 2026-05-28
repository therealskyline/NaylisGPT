import math
from typing import List, Union


class WSDScheduler:

    def __init__(
        self,
        optimizers: Union[object, List[object]],
        max_lr: float,
        total_steps: int,
        warmup_ratio: float = 0.03,
        decay_ratio: float = 0.15,
        min_lr_ratio: float = 0.1,
    ):
        self.optimizers   = optimizers if isinstance(optimizers, list) else [optimizers]
        self.max_lr       = max_lr
        self.min_lr       = max_lr * min_lr_ratio
        self.total_steps  = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.decay_steps  = int(total_steps * decay_ratio)
        self.stable_steps = total_steps - self.warmup_steps - self.decay_steps
        self.current_step = 0

    def get_lr(self) -> float:
        s = self.current_step
        if s < self.warmup_steps:
            return self.max_lr * (s / max(self.warmup_steps, 1))
        elif s < self.warmup_steps + self.stable_steps:
            return self.max_lr
        else:
            d = s - self.warmup_steps - self.stable_steps
            p = min(d / max(self.decay_steps, 1), 1.0)
            return self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * p))

    def step(self) -> float:
        lr = self.get_lr()
        self.current_step += 1
        for opt in self.optimizers:
            for pg in opt.param_groups:
                pg["lr"] = lr * 5.0 if pg.get("is_muon", False) else lr
        return lr

    def get_last_lr(self) -> List[float]:
        return [self.get_lr()]

    def state_dict(self) -> dict:
        return {"current_step": self.current_step}

    def load_state_dict(self, sd: dict):
        self.current_step = sd["current_step"]
