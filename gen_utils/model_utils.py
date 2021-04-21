import os
import torch
import numpy as np

from torch import nn
from models.unetvgg import UNetSimple
def get_saved_model(model, saved_model_dir, with_edge=True):
    """get saved model

    Parameters
    ----------
    model : `torch.module`
        model
    saved_model_dir : str
        dir of saved model
    with_edge : bool, optional
        whether edge traied model, by default True

    Returns
    -------
    `torch.module`
        model
    """
    saved_model = model#UNETvgg()
    for filename in os.listdir(saved_model_dir):
        file_path = os.path.join(saved_model_dir, filename)

        if os.path.isfile(file_path) and "epoch" in file_path and "edge" in file_path and with_edge:
            save_model_path = file_path
            break
        elif os.path.isfile(file_path) and "epoch" in file_path and "edge" not in file_path and not with_edge:
            save_model_path = file_path
            break
    if torch.cuda.is_available():
        checkpoint = torch.load(save_model_path)
    else:
        checkpoint = torch.load(save_model_path, map_location=torch.device('cpu'))    
    saved_model.load_state_dict(checkpoint['state_dict'])
    return saved_model

def pretrain_unet(tetra):
#pretrain unet with tetranet
    unet= UNetSimple(in_classes=1, channelscale=128)
    unet.conv1 = tetra.conv1
    unet.conv2 = tetra.conv2
    unet.conv3 = tetra.conv3
    unet.conv4 = tetra.conv4
    unet.conv5 = tetra.conv5
    unet.dec1 = tetra.dec1
    unet.dec2 = tetra.dec2
    unet.dec3 = tetra.dec3
    unet.dec4 = tetra.dec4
    unet.final = tetra.final
    return unet
