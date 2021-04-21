import torchvision
import matplotlib.pyplot as plt
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from gen_utils.model_utils import get_saved_model
from detect_region.pred_maskrcnn import test_pred
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
# load a model pre-trained pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# replace the classifier with a new one, that has
# num_classes which is user-defined
num_classes = 3  # 2 class (menisci) + background
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

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
# if your backbone returns a Tensor, featmap_names is expected to
# be [0]. More generally, the backbone should return an
# OrderedDict[Tensor], and in featmap_names you can choose which
# feature maps to use.
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                output_size=7,
                                                sampling_ratio=2)

# put the pieces together inside a FasterRCNN model
# model = FasterRCNN(backbone,
#                    num_classes=3,
#                    rpn_anchor_generator=anchor_generator,
#                    box_roi_pool=roi_pooler)



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
# import transforms as T

# def get_transform(train):
#     transforms = []
#     transforms.append(T.ToTensor())
#     if train:
#         transforms.append(T.RandomHorizontalFlip(0.5))
#     return T.Compose(transforms)
output_path = r"/home/students/thampi/PycharmProjects/MA_Praise/outputs"
data_loc = r"/home/students/thampi/PycharmProjects/meniscus_data/filt_data"#559 elements in filt data
hdf5file = r"/home/students/thampi/PycharmProjects/meniscus_data/segm.hdf5"
_, num_data_img, _, num_test_show, valid_after, no_epo, _, datafrom, in_chann, crop_size,_ , if_hdf5, batch_size, pretrained  = parse_helper()

no_epo, num_data_img, datafrom, if_hdf5, crop_size, batch_size=35, None, hdf5file, True, 512, 3
# num_test_show, valid_after, batch_size, in_chann = 20, None, 7, 4,3, 1

experiment_name = "maskrcnn_segm"#loss_dict[0]+loss_dict[1]+"_"+model_dict[0]
expt_outpath = os.path.join(output_path, experiment_name)
if not os.path.exists(expt_outpath):
    os.makedirs(expt_outpath)

def collate_fn(batch):
    return tuple(zip(*batch))
dataset = MeniscusDataForDetect(datafrom, num_img=num_data_img, in_channels=in_chann, #if_crop=True,
crop_size=crop_size,
if_hdf5=if_hdf5,
# is_crop_random=True,
                                    # if_aug=True,
                                    # only_cm=True
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
#below step is essential to make sure that all values in target dict are in device(cpu/cuda)
# targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
tar = []
# for ind in range(len(images)):
#     eachdict = {}
#     for key in list(targets.keys()):
#         eachdict.update({key:targets[key][ind]})
#     tar.append(eachdict)
# # targets = [{k: v for k, v in t.items()} for t in targets]

# output = model(images,tar)   # Returns losses and detections
# # For inference
# model.eval()
# x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
# predictions = model(x)

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

model = get_instance_segmentation_model(num_classes=3)

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
loss_hist = Averager()
itr = 1
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = None
#https://github.com/pytorch/vision/blob/master/references/detection/engine.py

# for epoch in range(num_epochs):
#     loss_hist.reset()

#     for images, targets, image_ids in data_loader:
#         images_out = []
#         for image in images:
#             images_out.append(image.to(device))
#         # images = list(image.to(device) for image in images)
#         #use collate_fn in dataloader
#         for t in targets:
#             for k,v in t.items():
#                 v=v.to(device)
#         # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#         #         print(targets)
#         #         break
#         # tar = []
#         # for ind in range(len(images)):
#         #     eachdict = {}
#         #     for key in list(targets.keys()):
#         #         eachdict.update({key: targets[key][ind].to(device)})
#         #     tar.append(eachdict)
#         loss_dict = model(images_out, targets)

#         losses = sum(loss for loss in loss_dict.values())
#         loss_value = losses.item()

#         loss_hist.send(loss_value)

#         optimizer.zero_grad()
#         losses.backward()
#         optimizer.step()

#         if itr % 50 == 0:
#             print(f"Iteration #{itr} loss: {loss_value}")

#         itr += 1

#     # update the learning rate
#     if lr_scheduler is not None:
#         lr_scheduler.step()

#     print(f"Epoch #{epoch} loss: {loss_hist.value}")
def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    # metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

model = model.to(device)
val_loader = torch.utils.data.DataLoader(
 val_set, batch_size=batch_size,#, shuffle=True)
#  , num_workers=4,
 collate_fn=collate_fn)

model = train_general(train_set, expt_outpath, expt_name=experiment_name, val_set_perc=0.15, test_set_perc=0, device=device,
                  num_epochs=no_epo, is_naive_train=True, mode="ele", val_set=val_set, test_set=None, valid_after=valid_after, 
                  net_model=model, batch_size=batch_size)

test_loader = torch.utils.data.DataLoader(
 test_set, batch_size=batch_size,#, shuffle=True)
#  , num_workers=4,
 collate_fn=collate_fn)
x, y , pred_segm = test_pred(model, test_loader, experiment_name=experiment_name,
loss_func=None, sgm_train=True)
save_pred_one(x, y, pred_segm, expt_outpath, bat_one=False, fig_name="res", nosft=True, channelcnt=3)
save_test_img_grid(x, y, pred_segm, expt_outpath, nosft=True, channel_cnt=3, fig_name=experiment_name)
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



