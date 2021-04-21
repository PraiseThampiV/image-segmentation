import random
import os
import numpy as np
import matplotlib.pyplot as plt
from loss.dice import dice_coefficient
from gen_utils.img_utils import get_roinp
def save_pred_act(x, y, pred_sgm, saved_res_dir, fig_name="res", dice=None):
    """save images with wrong predictions as figures

    Parameters
    ----------
    x : `torch.tensor`
        input
    y : `torch.tensor`
        target
    pred_sgm : `torch.tensor`
        predictions
    saved_res_dir : str
        save location
    fig_name : str, optional
        figure name, by default "res"
    dice : `torch.tensor`, optional
        loss, by default None
    """
    all_ = [item.clone().detach().cpu().numpy() for item in [x, y, pred_sgm, dice]]
    x, y, pred_sgm, dice = all_
    if y.shape[1]==3:
        y = np.argmax(y,axis=1)
    if len(pred_sgm.shape)==3:
        pred_sgm = np.argmax(pred_sgm,axis=-3)
    y=y.squeeze()

    # dice_coef = dice_coefficient(pred_sgm[img_num].view(1,1,pred_sgm.shape[-1],pred_sgm.shape[-1]), y[img_num].unsqueeze(dim=0))
    plt.tight_layout()
    fig_img, ax = plt.subplots(1,4)
    fig_img.set_size_inches(24,10)
    ax[0].set_title('Image')
    ax[0].axis("off")
    ax[0].imshow(x[0,:,:])
    ax[1].set_title('Actual Segmentation Mask')
    ax[1].axis("off")
    ax[1].imshow(y[:,:])
    ax[2].set_title(f'Predicted Mask, dice score:{np.round(dice, 2)}')
    ax[2].axis("off")
    ax[2].imshow(pred_sgm[:,:])
    ax[3].set_title(f'Predicted Mask and actual, dice score:{np.round(dice, 2)}')
    ax[3].axis("off")
    # find incorrect pixels, 3 for incorr fg, 4 for incorr bg
    comp = np.zeros_like(y)
    comp[y!=pred_sgm]=4
    comp[np.logical_and(pred_sgm==1,y==1)] = 1
    comp[np.logical_and(y==pred_sgm, y==2)] = 2
    comp[np.logical_and(y!=pred_sgm, y==0)] = 3
    xmin, ymin, xmax, ymax = get_roinp(y)
    im3 = ax[3].imshow(comp[ymin:ymax, xmin:xmax])
    fig_img.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)
    plt.savefig(os.path.join(saved_res_dir,f"{fig_name}.png"), bbox_inches = 'tight')
    plt.close("all")


