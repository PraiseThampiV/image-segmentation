import os
import torch
from torch import nn
from data_read.read_3dinput import Meniscus3DMidSlice
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

    data_loc = r"/home/students/thampi/PycharmProjects/meniscus_data/3d_inputs_all_slices"
    output_path = r"/home/students/thampi/PycharmProjects/MA_Praise/outputs"
    # add keys according to your experiment
    experiment_name = "train_midlsl"#loss_dict[0]+loss_dict[1]+"_"+model_dict[0]
    expt_outpath = os.path.join(output_path, experiment_name)
    if not os.path.exists(expt_outpath):
        os.makedirs(expt_outpath)

    no_epo, num_data_img, num_test_show, valid_after, in_channels, lr, batch_size = 2000, None, 1, 4, 1, 1e-5, 12
    dataset = Meniscus3DMidSlice(data_loc, num_img=num_data_img, in_channels=in_channels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DetectMid(in_classes=in_channels)
    loss_criterion = [nn.CrossEntropyLoss()]
    train_general(dataset, expt_outpath, expt_name=experiment_name, val_set_perc=0.15, device=device,
                num_epochs=no_epo, is_naive_train=True, valid_after=valid_after, net_model=model, batch_size=batch_size, test_set_perc=0.15,
                loss_criterion=loss_criterion, val_acc_func=nn.CrossEntropyLoss(), sgm_train=False, 
                test_batch_size=num_test_show, lr=lr)