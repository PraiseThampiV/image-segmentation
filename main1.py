import os
import torch

from torch.utils.data import DataLoader
from dlutils.losses.focal import focal_loss_with_logits
from models.unetvgg import UNetVgg
from data_read.meniscus_data import MeniscusData
from pl_utils.pl_train import train_model
from pl_utils.pl_mod import SegmModelPL
from loss.dice import dice_coefficient, dice_loss_sq
from pl_utils.pl_train import train_model
from result_utils.save_img import save_pred_one, test_pred, save_test_img_grid
from pl_utils.pl_test import test_model
from pytorch_lightning.logging.neptune import NeptuneLogger
from semeda_train.res_save import get_saved_model
from data_read.meniscus_data import MeniscusDataFilt

def meniscus_segm(data_loc, out_path, expt_name=None, train_set_perc=0.75, val_set_perc=0.15, device=None):
    full_data = MeniscusData(data_loc, num_img=20)
    train_size = int(train_set_perc * len(full_data))
    val_size = int(val_set_perc * len(full_data))
    test_size = len(full_data) - train_size - val_size
    train_set, val_set, test_set = torch.utils.data.random_split(full_data, [train_size, val_size, test_size])
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    htc_gpu = torch.cuda.device_count() if device.type=='cuda' else 0
    kwargs = {'num_workers': 4*htc_gpu, 'pin_memory': True} if device == 'cuda' else {}

    net_model = UNetVgg()
    batch_size, test_batch_size = 3, 4
    lightn_mod = SegmModelPL(net_model, train_set, val_set, test_set, batch_size, test_batch_size,
    loss_criterion=[dice_loss_sq, focal_loss_with_logits], val_acc_func=dice_coefficient,experiment_name=expt_name)
    lightn_mod=lightn_mod.to(device)
    trainer = train_model(lightn_mod, out_path, del_pre_files=True, device=device, num_epochs=1)

    test_model(lightn_mod, out_path, trainer)

    # saving results
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, **kwargs)
    img, segm, pred = test_pred(lightn_mod, out_path, test_loader, expt_name, trainer)
    save_pred_one(img, segm, pred, out_path)
    save_test_img_grid(img, segm, pred, out_path)

if __name__=="__main__":
    loss_dict = {0:"dice", 1:"focal"}
    model_dict = {0:"vggunet"}
    data_loc = r"/home/students/thampi/PycharmProjects/meniscus_segm/data"
    output_path = r"/home/students/thampi/PycharmProjects/meniscus_segm/outputs"
    # add keys according to your experiment
    experiment_name = loss_dict[0]+loss_dict[1]+"_"+model_dict[0]
    expt_outpath = os.path.join(output_path, experiment_name)
    if not os.path.exists(expt_outpath):
        os.makedirs(expt_outpath)
    meniscus_segm(data_loc, expt_outpath, experiment_name)
