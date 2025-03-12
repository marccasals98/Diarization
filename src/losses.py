# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Copyright 2022 Brno University of Technology (authors: Federico Landini, Lukas Burget, Mireia Diez)
# Copyright 2022 AUDIAS Universidad Autonoma de Madrid (author: Alicia Lozano-Diez)
# Copyright 2025 Barcelona Supercomputing Center (author: Marc Casals i Salvador)

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple
from torch.nn.functional import logsigmoid

def pit_loss_multispk(logits: List[torch.Tensor],
                    target: List[torch.Tensor],
                    n_speakers:np.ndarray,
                    detach_attractor_loss: bool)->torch.Tensor:
    if detach_attractor_loss:
        ...