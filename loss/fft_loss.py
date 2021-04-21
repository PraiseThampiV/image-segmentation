
import torch
from torch import nn
from loss.edge_loss import get_hot_enc
def fft_loss_logits(pred, segm):
    """mse between fft 

    Parameters
    ----------
    pred : [type]
        predicted segmentation mask,(before applying softmax2d)
    segm : [type]
        actual mask with labels, (not one hot encoded)

    Returns
    -------
    `torch.tensor`
        loss
    """
    sft = nn.Softmax2d()
    #TODO try withnormalized otpion of rfft
    try:
        fft_segm = torch.rfft(get_hot_enc(segm),2, onesided=False)
        fft_pred = torch.rfft(sft(pred),2, onesided=False)
    except RuntimeError as e:
        import torch.fft as tfft
        fft_segm = tfft.rfft(get_hot_enc(segm),dim=2)
        fft_pred = tfft.rfft(sft(pred),dim=2)
    mse = nn.MSELoss()
    return mse(fft_segm, fft_pred)
