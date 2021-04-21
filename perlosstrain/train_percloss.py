import os
import sys
import torch

from models.unetvgg import UNetSimple, SpiderNet
from data_read.meniscus_data import MeniscusDataAllOptions
from perlosstrain.common import train_general, train_whole_img

def deep_sup(param_list):
    """training by deep supervision

    Parameters
    ----------
    param_list : list
        parameter list
    """
    torch.cuda.empty_cache()
    seed = 20
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    output_path = r"/home/students/thampi/PycharmProjects/MA_Praise/outputs"

    no_epo, num_data_img, test_ind_start_img, num_test_show, valid_after, no_epo1, _, datafrom, in_channels, cropsize, experiment_name, if_hdf5, batch_size, pretrained, _ = param_list

    expt_outpath = os.path.join(output_path, experiment_name)
    if not os.path.exists(expt_outpath):
        os.makedirs(expt_outpath)
    test_dice_coeff=0
    hdf5file = datafrom
    data_set = MeniscusDataAllOptions(hdf5file, num_img=num_data_img,
    crop_size=cropsize,
    if_hdf5=if_hdf5,
    in_channels=in_channels, 
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SpiderNet(in_classes=in_channels, multiout=True, channelscale=128)
    train_general(dataset=data_set, out_path=expt_outpath, expt_name=experiment_name, val_set_perc=0.15, test_set_perc=.15, 
    device=device,num_epochs=no_epo, is_naive_train=True, valid_after=valid_after, net_model=model, batch_size=batch_size, test_batch_size = 3)
