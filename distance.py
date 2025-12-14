import numpy as np
import torch
import torch.nn as nn

def repulsion_loss(points, h=0.03):
    # points: (B, N, 3)
    dist = torch.cdist(points, points)
    mask = (dist > 0).float()
    return torch.mean(torch.exp(-dist * dist / h) * mask)


class ChamferLoss(nn.Module):
    """Symmetric Chamfer distance for batched point clouds.

    Accepts `target_pc` and `output_pc` shaped `(B, M, D)` and `(B, N, D)`.
    Returns either a scalar (reduction='mean' or 'sum') or per-batch losses
    (reduction='none').
    """

    def __init__(self, squared: bool = False, eps: float = 1e-12, reduction: str = 'mean'):
        super(ChamferLoss, self).__init__()
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.squared = squared
        self.eps = eps
        self.reduction = reduction

    def forward(self, target_pc: torch.Tensor, output_pc: torch.Tensor) -> torch.Tensor:
        target = target_pc.float()
        output = output_pc.float()

        # compute pairwise squared distances per batch: (B, N, M)
        if self.squared:
            # use algebraic expansion for squared distances
            xx = torch.sum(output * output, dim=2, keepdim=True)  # (B, N, 1)
            yy = torch.sum(target * target, dim=2, keepdim=True)  # (B, M, 1)
            cross = -2.0 * torch.bmm(output, target.transpose(1, 2))  # (B, N, M)
            pairwise_sq = cross + xx + yy.transpose(1, 2)
            pairwise_sq = torch.clamp(pairwise_sq, min=0.0)
            dists = pairwise_sq
        else:
            # compute squared distances then sqrt for Euclidean distances
            xx = torch.sum(output * output, dim=2, keepdim=True)  # (B, N, 1)
            yy = torch.sum(target * target, dim=2, keepdim=True)  # (B, M, 1)
            cross = -2.0 * torch.bmm(output, target.transpose(1, 2))  # (B, N, M)
            pairwise_sq = cross + xx + yy.transpose(1, 2)
            pairwise_sq = torch.clamp(pairwise_sq, min=0.0)
            dists = torch.sqrt(pairwise_sq + self.eps)

        # for each output point, distance to nearest target point -> (B, N)
        mins_output_to_target = torch.min(dists, dim=2)[0]
        loss_a_to_b = torch.sum(mins_output_to_target, dim=1)  # (B,)

        # for each target point, distance to nearest output point -> (B, M)
        mins_target_to_output = torch.min(dists, dim=1)[0]
        loss_b_to_a = torch.sum(mins_target_to_output, dim=1)  # (B,)

        per_sample_loss = loss_a_to_b + loss_b_to_a / 0.5  # (B,)

        if self.reduction == 'none':
            return per_sample_loss
        elif self.reduction == 'sum':
            return torch.sum(per_sample_loss)
        else:  # mean
            return torch.mean(per_sample_loss)


if __name__ == "__main__":
    a = torch.tensor([[0., 2.], [0., 5.], [0., 7.]], requires_grad=False)
    b = torch.tensor([[1., 2.], [10., 5.], [2., 7.]], requires_grad=True)
    loss_fn = ChamferLoss()
    loss = loss_fn(a, b)
    print('loss:', loss.item())
    loss.backward()
    print('grad on b[0]:', b.grad)