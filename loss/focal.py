import torch
from rising.transforms.functional.channel import one_hot_batch
from torch.nn import functional as F
#adapted from https://github.com/justusschock/dl-utils
def reduce(to_reduce: torch.Tensor, reduction: str) -> torch.Tensor:
    """
    reduces a given tensor by a given reduction method
    Parameters
    ----------
    to_reduce : torch.Tensor
        the tensor, which shall be reduced
    reduction : str
        a string specifying the reduction method.
        should be one of 'elementwise_mean' | 'none' | 'sum'
    Returns
    -------
    torch.Tensor
        reduced Tensor
    Raises
    ------
    ValueError
        if an invalid reduction parameter was given
    """
    if reduction == 'elementwise_mean':
        return torch.mean(to_reduce)
    if reduction == 'none':
        return to_reduce
    if reduction == 'sum':
        return torch.sum(to_reduce)
    raise ValueError('Reduction parameter unknown.')

def _general_focal_loss(p: torch.Tensor, t: torch.Tensor, gamma: float,
                        loss_val: torch.Tensor, alpha_weight: float = 1.,
                        reduction: str = 'elementwise_mean'):
    """
    Helper Function Handling the general focal part and the reduction
    Parameters
    ----------
    p: torch.Tensor
        the prediction tensor
    t : torch.Tensor
        the target tensor
    gamma : float
        focusing parameter
    loss_val : torch.Tensor
        the value coming from the previous loss function
    alpha_weight : float
        class weight
    reduction : str
        reduction parameter
    Returns
    -------
    torch.Tensor
        loss value
    Raises
    ------
    ValueError
        invalid reduction parameter
    """
    # compute focal weights
    # if not isinstance(alpha_weight, torch.Tensor):
    #     alpha_weight = torch.tensor([1.], device=p.device)

    focal_weight = 1 - torch.where(torch.eq(t, 1.), p, 1 - p)
    focal_weight.pow_(gamma)
    focal_weight.to(p.device)

    # adjust shape if necessary
    if len(loss_val.shape) < len(focal_weight.shape):
        loss_val = loss_val.unsqueeze(1)

    # compute loss
    focal_loss = focal_weight * alpha_weight * loss_val

    return reduce(focal_loss, reduction)

def _focal_loss(p: torch.Tensor, t: torch.Tensor, gamma: float,
                loss_val: torch.Tensor, reduction: str):
    """
    Focal loss helper function
    Parameters
    ----------
    p: torch.Tensor
        the prediction tensor
    t : torch.Tensor
        the target tensor
    gamma : float
        focusing parameter
    loss_val : torch.Tensor
        value coming from the previous (weighted) loss function
    reduction : str
        reduction parameter
    Returns
    -------
    torch.Tensor
        loss value
    Raises
    ------
    ValueError
        invalid reduction parameter
    """
    n_classes = p.size(1)
    target_onehot = one_hot_batch(t.unsqueeze(1), num_classes=n_classes)
    return _general_focal_loss(p=p, t=target_onehot, gamma=gamma,
                               loss_val=loss_val, reduction=reduction,
                               alpha_weight=1.)
                               
def focal_loss_with_logits(p: torch.Tensor, t: torch.Tensor, gamma: float = 2.,
                           alpha: torch.Tensor = None,
                           reduction: str = 'elementwise_mean'):
    """
    focal loss with logits
    Parameters
    ----------
    p: torch.Tensor
        the prediction tensor
    t : torch.Tensor
        the target tensor
    gamma : float
        focusing parameter
    alpha : torch.Tensor
        class weight
    reduction : str
        reduction parameter
    Returns
    -------
    torch.Tensor
        loss value
    Raises
    ------
    ValueError
        invalid reduction parameter
    """
    loss_val = F.cross_entropy(p, t, weight=alpha, reduction='none')
    p = F.softmax(p, dim=1)
    return _focal_loss(p=p, t=t, gamma=gamma, reduction=reduction,
                       loss_val=loss_val)