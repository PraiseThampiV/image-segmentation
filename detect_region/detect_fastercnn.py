#adapted from #https://github.com/pytorch/vision/blob/master/references/detection/engine.py
import torchvision
import matplotlib.pyplot as plt
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from gen_utils.model_utils import get_saved_model
from detect_region.pred_box_segm import test_pred
from models.unetvgg import UNetSimple
from result_utils.save_img import save_test_img_grid, save_pred_one
from detect_region.common import train_general
from gen_utils.model_utils import get_saved_model
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from gen_utils.main_utils import parse_helper
import os
import torch
from detect_region.read_data import MeniscusDataForDetect
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
def collate_fn(batch):
    return tuple(zip(*batch))

class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model
    
def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

def check_target():
    from matplotlib import patches
    f, a = plt.subplots()
    im_ind=1
    im = a.imshow(images[im_ind][0].detach().cpu())
    labind = 1
    box = targets[im_ind]['boxes'][labind]
    rect = patches.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],linewidth=1,edgecolor='r',facecolor='none')
    a.add_patch(rect)
    plt.savefig(r"/home/students/thampi/PycharmProjects/MA_Praise/outputs/check.png")
    a.imshow(targets[im_ind]['masks'].clone().cpu().numpy()[labind])
    plt.savefig(r"/home/students/thampi/PycharmProjects/MA_Praise/outputs/check.png")
    
def segm_fasterrcnn(param_list):  
    """detect roi by faster rcnn then segm

    Parameters
    ----------
    param_list : list
        parameters for training
    """
    #segmentation after detecting ROI using Faster R-CNN  
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = 3  # 2 class (menisci) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # load a pre-trained model for classification and return
    # only the features
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    # FasterRCNN needs to know the number of
    # output channels in a backbone. For mobilenet_v2, it's 1280
    # so we need to add it here
    backbone.out_channels = 1280

    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                    aspect_ratios=((0.5, 1.0, 2.0),))

    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                    output_size=7,
                                                    sampling_ratio=2)

    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone,
                    num_classes=3,
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler)


    output_path = r"/home/students/thampi/PycharmProjects/MA_Praise/outputs"
    data_loc = r"/home/students/thampi/PycharmProjects/meniscus_data/filt_data"#559 elements in filt data
    hdf5file = r"/home/students/thampi/PycharmProjects/meniscus_data/segm.hdf5"
    _, num_data_img, _, num_test_show, valid_after, no_epo, _, datafrom, in_chann, crop_size,experiment_name, if_hdf5, batch_size, pretrained, _  = param_list

    # experiment_name = "fasterrcnn_test"
    expt_outpath = os.path.join(output_path, experiment_name)
    if not os.path.exists(expt_outpath):
        os.makedirs(expt_outpath)


    dataset = MeniscusDataForDetect(datafrom, num_img=num_data_img, in_channels=in_chann, #if_crop=True,
    crop_size=crop_size,
    if_hdf5=if_hdf5,
                                        )
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    data_len = len(dataset)
    ts, vs = int(0.8*data_len), int(0.1*data_len)#training set size, validation set size
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [ts, vs, data_len-(ts+vs)])
    data_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size,#, shuffle=True)
    #  , num_workers=4,
    collate_fn=collate_fn)
    # For Training
    images,targets,index = next(iter(data_loader))
    images = list(image for image in images)

    tar = []

    loss_hist = Averager()
    itr = 1
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = None


    model = model.to(device)
    val_loader = torch.utils.data.DataLoader(
    val_set, batch_size=batch_size,#, shuffle=True)
    #  , num_workers=4,
    collate_fn=collate_fn)

    model = train_general(train_set, expt_outpath, expt_name=experiment_name, val_set_perc=0.15, test_set_perc=0, device=device,
                    num_epochs=no_epo, is_naive_train=True, mode="ele", val_set=val_set, test_set=None, valid_after=valid_after, 
                    net_model=model, batch_size=batch_size)
    model = get_saved_model(model, expt_outpath, with_edge=False)

    model2_path = os.path.join(r"/home/students/thampi/PycharmProjects/MA_Praise/outputs", "segm_roi")
    model2 = UNetSimple(in_classes=1, channelscale=128, out_classes=2)
    model2 = get_saved_model(model2, model2_path, with_edge=False)
    test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=batch_size,#, shuffle=True)
    #  , num_workers=4,
    collate_fn=collate_fn)
    x, y , pred_segm = test_pred(model, model2, test_loader, experiment_name="",
    loss_func=None, sgm_train=True, crop_size=crop_size, allow_faulty=True)
    save_pred_one(x, y, pred_segm, expt_outpath, bat_one=False, fig_name="res", nosft=True, channelcnt=3)
    save_test_img_grid(x, y, pred_segm, expt_outpath, nosft=True, channel_cnt=3, fig_name=experiment_name)




