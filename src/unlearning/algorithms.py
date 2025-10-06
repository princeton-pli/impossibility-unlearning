from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class UnlearnConfig:
    algorithm: str = "GA"         # GA | NPO | SimNPO
    steps: int = 10
    learning_rate: float = 1e-5
    beta: float = 2.0             # NPO/SimNPO
    reference_model_path: Optional[str] = None

class UnlearningAlgorithm:
    def __init__(self, cfg: UnlearnConfig):
        self.cfg = cfg

    def step(self, batch) -> Dict[str, Any]:
        """Placeholder for one unlearning update on a batch.
        Replace with your trainer/framework of choice.
        """
        # TODO: implement GA/NPO/SimNPO updates
        return {"loss": 0.0}

class GA(UnlearningAlgorithm):
    pass

class NPO(UnlearningAlgorithm):
    pass

class SimNPO(UnlearningAlgorithm):
    pass
