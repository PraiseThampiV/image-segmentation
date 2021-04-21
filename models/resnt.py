import torch
import torch.nn.functional as F
import torchvision.models as models
from torch import nn
from models.resnetdeep import generate_model as rg
from models.densenet3d import generate_model as dg

class DetectMid(nn.Module):
    def __init__(self, pretrained=False, in_classes=1, out_classes=1, patch_train=False, if_convtrans=False, if_regr=False):
        """network to detect middle slice

        Parameters
        ----------
        pretrained : bool, optional
            if pretrained, by default False
        out_classes : int, optional
            output channel count, by default 3
        patch_train : bool, optional
            if patches are input, by default False
        if_convtrans : bool, optional
            if transposed convolution to be used, by default False
        if_regr : bool, by default False
            whether as classification or as regression output
        """
        super(DetectMid, self).__init__()
        self.rsnet = dg(169)
        #models.video.r3d_18(pretrained=True, progress=False)
        self.patch_train=False

        self.first = nn.Conv3d(in_classes, 3, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        if self.rsnet==rg:
            self.last = nn.Linear(400,80)
        else:
            self.last = nn.Linear(1000,80)
        self.lastregr = nn.Linear(80, out_classes)#last regression layer
        self.if_regr = if_regr

    def forward(self, x):
        out = self.first(x)
        out = self.rsnet(out)
        out = self.last(out)
        if self.if_regr:
            out = self.lastregr(out)
            out = out.squeeze(dim=-1)# to avoid warning of being in diff shape by MSEloss
        return out


class DetectSide(nn.Module):
    def __init__(self, pretrained=False, in_classes=1, out_classes=1, patch_train=False, if_convtrans=False, if_regr=False):
        """network to detect middle slice

        Parameters
        ----------
        pretrained : bool, optional
            if pretrained, by default False
        out_classes : int, optional
            output channel count, by default 3
        patch_train : bool, optional
            if patches are input, by default False
        if_convtrans : bool, optional
            if transposed convolution to be used, by default False
        if_regr : bool, by default False
            whether as classification or as regression output
        """
        super(DetectSide, self).__init__()
        self.mod = models.resnet18(pretrained=False)
        #models.video.r3d_18(pretrained=True, progress=False)
        self.patch_train=False

        self.first = nn.Conv2d(in_classes, 3, kernel_size=3, stride=1, padding=1, bias=False)
        # if self.mod==rg:
        self.last = nn.Linear(1000,80)
        # else:
            # self.last = nn.Linear(1000,80)
        self.lastregr = nn.Linear(80, out_classes)#last regression layer
        self.if_regr = if_regr

    def forward(self, x):
        out = self.first(x)
        out = self.mod(out)
        out = self.last(out)
        if self.if_regr:
            out = self.lastregr(out)
            out = out.squeeze(dim=-1)# to avoid warning of being in diff shape by MSEloss
        return out
