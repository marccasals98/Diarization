# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Copyright 2022 Brno University of Technology (authors: Federico Landini, Lukas Burget, Mireia Diez)
# Copyright 2022 AUDIAS Universidad Autonoma de Madrid (author: Alicia Lozano-Diez)
# Copyright 2025 Barcelona Supercomputing Center (author: Marc Casals i Salvador)

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple
from torch.nn.functional import logsigmoid
from torch.nn.modules.loss import _Loss
from scipy.optimize import linear_sum_assignment
import ipdb



def pit_loss_multispk(logits: List[torch.Tensor],
                    target: List[torch.Tensor],
                    n_speakers:np.ndarray,
                    detach_attractor_loss: bool)->torch.Tensor:
    """
    # PIT Loss for multi-speaker diarization.
    
    ## Motivation
    We want to solve the permutation problem in the PIT Loss for multi-speaker diarization.
    [[1,0],[0,1]] and [[0,1],[1,0]] are the same but the loss will be different. So we consider
    all the possible permutations and choose the one that minimizes the loss.

    Args:
        logits (List[torch.Tensor]): The logits generated by the model. The shape is [batch_size, n_frames, n_speakers]. HACK Now the shape is [batch_size, n_speajers] because it is not correct.
        target (List[torch.Tensor]): The target labels. The shape is [batch_size, n_frames, n_speakers].
        n_speakers (np.ndarray): List with the number of speakers in each...
        detach_attractor_loss (bool): If True, the attractor loss is detached from the rest of the network?

    Returns:
        torch.Tensor: _description_
    """
    target = target.float()
    if detach_attractor_loss:
        # -1s for speakers that do not have valid attractor
        # HACK Is this shape correct? This is the batch size.
        for i in range(target.shape[0]):
            target[i, :, n_speakers[i]:] = -1 * torch.ones(target.shape[1], target.shape[2] - n_speakers[i])
    
    logits_t = logits.detach().transpose(1, 2)

    cost_matrices = -logsigmoid(logits_t).bmm(target) - logsigmoid(-logits_t).bmm(1-target)

    max_n_speakers = max(n_speakers)

    # Compute all the possible permutations
    for i, cost_matrice in enumerate(cost_matrices):
        if max_n_speakers > n_speakers[i]:
            # sum all absolute values of the cost matrix
            max_value = torch.max(torch.abs(cost_matrice))

            # We assign the max value to the new speakers that we have added
            cost_matrices[-(max_n_speakers-n_speakers[i]):] = max_value
            cost_matrices[:, -(max_n_speakers-n_speakers[i]):] = max_value
        
        # Compute the linear sum assignment
        # rows are the predictions, columns are the references.
        pred_alig, ref_alig = linear_sum_assignment(cost_matrice.cpu().numpy())

        # Assert if the 
        assert (np.all(pred_alig == np.arange(logits.shape[-1])))
        # This is the permutation that minimizes the loss
        target[i, :] = target[i, :, ref_alig]
        
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")

    # We set the loss to 0 for the speakers that do not have valid attractor
    loss[torch.where(target == -1)] = 0

    # Normalize by sequence length
    loss = torch.sum(loss, axis=1) / (target != -1).sum(axis=1)
    for i in range(loss.shape[0]):
        loss[i, n_speakers[i]:] = torch.zeros(loss.shape[1] - n_speakers[i])
    
    # Normalize in batch for all speakers.
    loss = torch.mean(loss)
    return loss

class PITLoss(_Loss):

    def __init__(self, n_speakers, detach_attractor_loss, size_average=None, reduce=None, reduction = "mean"):
        super().__init__(size_average, reduce, reduction)
        self.n_speakers = n_speakers
        self.detach_attractor_loss = detach_attractor_loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, n_speakers: np.ndarray)->torch.Tensor:
        """Implements the PIT Loss for multi-speaker diarization. 

        Args:
            input (torch.Tensor): The logits generated by the model
            target (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        return pit_loss_multispk(input, target, n_speakers, self.detach_attractor_loss)