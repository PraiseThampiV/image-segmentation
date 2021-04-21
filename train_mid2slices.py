
import os
import torch
import argparse
from torch import nn
from data_read.read_3dinput import Meniscus3DMidSlice2
from models.resnt import DetectMid
from train_helper.common import train_general
if __name__=="__main__":
    torch.cuda.empty_cache()
    import gc 
    gc.collect() 
    seed = 20
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data_loc = r"/home/students/thampi/PycharmProjects/meniscus_data/3d_inputs_all_slices_whole"
    output_path = r"/home/students/thampi/PycharmProjects/MA_Praise/outputs"
    # add keys according to your experiment
    experiment_name = "train_mid_dess_slices"#loss_dict[0]+loss_dict[1]+"_"+model_dict[0]
    expt_outpath = os.path.join(output_path, experiment_name)
    if not os.path.exists(expt_outpath):
        os.makedirs(expt_outpath)

    no_epo, num_data_img, num_test_show, valid_after, in_channels, lr, batch_size, downscale = 300,None, 1, 4, 1, 1e-4, 1, 0.25
    parser = argparse.ArgumentParser(description='List the content of a folder')
    parser.add_argument('-batch_size', metavar='batch size', type=int, help='batch size', default=batch_size)
    args = parser.parse_args()
    batch_size = args.batch_size
    dataset = Meniscus3DMidSlice2(data_loc, num_img=num_data_img, in_channels=in_channels, downscale=downscale)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DetectMid(in_classes=in_channels, if_regr=True, out_classes=2)
    loss_criterion = [nn.MSELoss()]
    train_general(dataset, expt_outpath, expt_name=experiment_name, val_set_perc=0.15, device=device,
                num_epochs=no_epo, is_naive_train=True, valid_after=valid_after, net_model=model, batch_size=batch_size, test_set_perc=0.15,
                loss_criterion=loss_criterion, val_acc_func=nn.MSELoss(), sgm_train=False, 
                test_batch_size=num_test_show, lr=lr)