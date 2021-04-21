import torch
from torch import nn
from collections import OrderedDict
from models.unetvgg import UNetVgg, UNetSimple
import torch.nn.functional as F

class DetectPatchAndSegm(nn.Module):
    def __init__(self, in_classes=1, channelscale=64):
        """model to output patch and masks

        Parameters
        ----------
        in_classes : int, optional
            in channels, by default 1
        channelscale : int, optional
            channel factor, by default 64
        """
        super(DetectPatchAndSegm, self).__init__()
        self.unetvgg1 = UNetSimple(in_classes=in_classes, channelscale=64, out_classes=2)#UNetVgg()
        self.unetvgg2 = UNetSimple(in_classes=3, channelscale=128, out_classes=3)#in is 2 patches and original image
        self.sft = nn.Softmax2d()

    def forward(self, x):
        patch_imgs = self.unetvgg1(x)
        in2 = torch.cat([patch_imgs, x], dim=-3)
        output = self.unetvgg2(in2)
        # ttt=cycle_out[0],cycle_out[1],cycle_out[2],cycle_out[-1],cycle_out[3], cycle_out[4], cycle_out[5], cycle_out[6]
        return output, patch_imgs