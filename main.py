from functools import partial
from gen_utils.main_utils import parse_helper
from train_helper.train_baseline import train_baseline
from train_helper.train_classify import train_classify
from detect_region.detect_fastercnn import segm_fasterrcnn
from findpatsegm.findps_train import find_pat_masks
from semeda_train.train_semeda_bp import train_edge_aware
from aggr_stratgy.train_patch_init import train_pat_init
from aggr_stratgy.train_patchextraggr import extragg
from perlosstrain.train_percloss import deep_sup
from roi_train.train_roi import two_unets
from tetranet.train_tetranet import combined_unets
import torch
if __name__=="__main__":
    import gc 
    gc.collect()
    torch.cuda.empty_cache()
    param_list = parse_helper()
    semeda_tra = partial(train_edge_aware, is_semeda=True)
    cycle_tra = partial(train_edge_aware, is_semeda=False)
    func_list = {"baseline":train_baseline, "faster_rcnn_segm":segm_fasterrcnn, "cascaded_unet":find_pat_masks, "semedanet":semeda_tra,"cycle_net":cycle_tra, "patch_init":train_pat_init,"extraggr":extragg, "deepsup":deep_sup,"unets_series":two_unets, "combined_unets": combined_unets, "train_side": train_classify}

    func_list[param_list[-1]](param_list)
