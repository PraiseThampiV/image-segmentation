#make two sets, train normally the roi regions using common methods., for testing use second set

import os
import torch
from torch import nn
from models.unetvgg import UNetVgg, UNetNaive, UNetWithoutCenter, UNetNaiveMultiOut, UNetSimple, UNetLimMultiOut, UNetSkip, SpiderNet
from roi_train.read_roi import ImageROI
from train_helper.common import train_general
from gen_utils.main_utils import parse_helper
from loss.dice import dice_loss_sq, dice_coefficient, dice_loss_sa, logcoshdiceloss
from loss.hausdorff import edge_conv_loss
# from roi_train.save_img import test_pred
from roi_train.save_img_2models import test_pred
from result_utils.save_img import save_pred_one, save_test_img_grid
from torch.utils.data import DataLoader
from data_read.meniscus_data import MeniscusDataAllOptions
from semeda_train.res_save import get_saved_model
def two_unets(param_list):
    torch.cuda.empty_cache()
    seed = 20
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    output_path =  r"/home/students/thampi/PycharmProjects/MA_Praise/outputs"
    hdf5file = r"/home/students/thampi/PycharmProjects/meniscus_data/segm.hdf5"#
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    htc_gpu = torch.cuda.device_count() if device.type=='cuda' else 0
    kwargs = {'num_workers': 4*htc_gpu, 'pin_memory': True} if device == 'cuda' else {}

    no_epo1, num_data_img, second_start, num_test_show, valid_after, no_epo2, _, datafrom, in_chann, crop_size, experiment_name, if_hdf5, batch_size,pretrained, _  = param_list
    allow_faulty=True
    test_batch_size = 4

    expt_outpath = os.path.join(output_path, experiment_name)
    if not os.path.exists(expt_outpath):
        os.makedirs(expt_outpath)

    model1 = UNetSimple(in_classes=1, channelscale=128)
    model1_path = os.path.join(output_path, "segm_dice1")
    if not os.path.exists(model1_path):
        os.makedirs(model1_path)
    wholeimgtrain_set = MeniscusDataAllOptions(datafrom, num_img=num_data_img, in_channels=in_chann, if_aug=False,
    crop_size=crop_size,
    if_hdf5=if_hdf5,part="first", second_start=second_start
                                      )
    train_general(wholeimgtrain_set, model1_path, val_set_perc=0.15, test_set_perc=0, device=device,
                num_epochs=no_epo1, is_naive_train=True, valid_after=valid_after, net_model=model1,batch_size=batch_size
                )
    model1 = get_saved_model(model1, model1_path, with_edge=False)

    model2 = UNetSimple(in_classes=1, channelscale=128, out_classes=2)
    model2_path = os.path.join(output_path, "segm_roi1")
    if not os.path.exists(model2_path):
        os.makedirs(model2_path)
    loss_criterion = [dice_loss_sq]#logcoshdiceloss]#dice_loss_sq, edge_conv_loss]#, mse]#
    data_set = ImageROI(datafrom, num_img=num_data_img, in_channels=in_chann, #if_crop=True,
    crop_size=crop_size,
    if_hdf5=if_hdf5,part="first", second_start=second_start
                                      )
    train_general(data_set, model2_path, val_set_perc=0.15, test_set_perc=0, device=device,
                num_epochs=no_epo2, is_naive_train=True, valid_after=valid_after, net_model=model2,batch_size=batch_size
                ,loss_criterion=loss_criterion)
    model2 = get_saved_model(model2, model2_path, with_edge=False)
    # saving results
    test_set = MeniscusDataAllOptions(datafrom, num_img=num_data_img, in_channels=in_chann, if_aug=False,
    crop_size=crop_size,
    if_hdf5=if_hdf5,part="second", second_start=second_start
                                      )
    # test_set = ImageROI(datafrom, num_img=num_data_img, in_channels=1, segm_layers=1, if_crop=False, crop_size=512, part=None, 
    # second_start=600, no_medial_only=True, if_hdf5=True, is_crop_random=False, if_aug=True, only_cm=False,
    # send_ends=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, **kwargs)
    # for testing model2, use test_pred from train_roi
    # img, segm, pred = test_pred(model2, model2_path, test_loader, experiment_name="", trainer=None, print_test_acc=True, loss_func=None, sgm_train=True)
    # save_pred_one(img, segm, pred, expt_outpath, bat_one=bat_one,fig_name="out1"+experiment_name,box=box, label=label, nosft=True)
    # save_test_img_grid(img, segm, pred, expt_outpath, fig_name="outs"+experiment_name,box=box, label=label, nosft=True)
    img, segm, pred = test_pred(model1, model2, test_loader, experiment_name, crop_size=crop_size, allow_faulty=allow_faulty)
    bat_one = True if len(segm.shape) == 3 and segm.shape[0] == 1 else False
    #below funcs from result_utils not from roi
    save_pred_one(img, segm, pred, expt_outpath, bat_one=bat_one,fig_name="out1"+experiment_name,nosft=True, channelcnt=3)
    save_test_img_grid(img, segm, pred, expt_outpath, fig_name="outs"+experiment_name,nosft=True, channel_cnt=3)

