from torch import nn
from collections import OrderedDict
from models.unetvgg import UNetVgg, UNetSimple
import torch.nn.functional as F


class EdgeNet(nn.Module):
    """Edge Net used for SEMEDA loss
    """
    def __init__(self):
        super(EdgeNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3,16,3, padding=1), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(16,32,3, padding=1), nn.ReLU())
        self.layer2_1 = nn.Sequential(nn.Conv2d(32,32,3, padding=1), nn.ReLU())
        self.layer2_2 = nn.Sequential(nn.Conv2d(32,32,3, padding=1), nn.ReLU())
        self.layer2_3 = nn.Sequential(nn.Conv2d(32,32,3, padding=1), nn.ReLU())
        self.layer2_4 = nn.Sequential(nn.Conv2d(32,32,3, padding=1), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(32,16,3, padding=1), nn.ReLU())
        self.layer4 = nn.Conv2d(16,3,3, padding=1)

    def forward(self, x):
        psi1 = self.layer1(x)
        psi2 = self.layer2(psi1)
        out_lay21 = self.layer2_1(psi2)
        out_lay22 = self.layer2_2(out_lay21)
        out_lay23 = self.layer2_3(out_lay22)
        out_lay24 = self.layer2_4(out_lay23)
        psi3 = self.layer3(out_lay24)
        psi4 = self.layer4(psi3)
        return psi1, psi2, out_lay21, out_lay22, out_lay23, out_lay24, psi3, psi4

class SemedaNet(nn.Module):
    def __init__(self, in_channels=1):
        super(SemedaNet, self).__init__()
        # self.unetvgg = UNetVgg()
        self.unet = UNetSimple(in_classes=in_channels)
        self.edgenet = EdgeNet()

    def forward(self, x):
        x = self.unet(x)#vgg(x)
        sft = nn.Softmax2d()
        x_map = sft(x)
        psi = self.edgenet.forward(x_map)
        return x, psi

class CycleNet(nn.Module):
    def __init__(self):
        super(CycleNet, self).__init__()
        #  border block
        channelnum = 16
        # self.blayer1 = nn.Sequential(nn.Conv2d(3,channelnum,3, padding=1), nn.InstanceNorm2d(channelnum))
        # self.blayer2 = nn.Sequential(nn.Conv2d(channelnum,channelnum,3, padding=1), nn.InstanceNorm2d(channelnum))
        # channelnum2 = 32
        # self.blayer2_1 = nn.Sequential(nn.Conv2d(channelnum,channelnum2,3, padding=1), nn.InstanceNorm2d(channelnum2))
        # self.blayer2_2 = nn.Sequential(nn.Conv2d(channelnum2,channelnum2,3, padding=1), nn.InstanceNorm2d(channelnum2))
        # self.blayer2_3 = nn.Sequential(nn.Conv2d(channelnum2,channelnum2,3, padding=1), nn.InstanceNorm2d(channelnum2))
        # self.blayer2_4 = nn.Sequential(nn.Conv2d(channelnum2,channelnum2,3, padding=1), nn.InstanceNorm2d(channelnum2))
        # self.blayer3 = nn.Sequential(nn.Conv2d(channelnum2,channelnum,3, padding=1), nn.InstanceNorm2d(channelnum))
        # self.blayer4 = nn.Sequential(nn.Conv2d(channelnum,2,3, padding=1))

        # #  mask block
        # self.mlayer1 = nn.Sequential(nn.Conv2d(2,channelnum,3, padding=1), nn.InstanceNorm2d(channelnum))
        # self.mlayer2 = nn.Sequential(nn.Conv2d(channelnum,channelnum,3, padding=1), nn.InstanceNorm2d(channelnum))
        # self.mlayer2_1 = nn.Sequential(nn.Conv2d(channelnum,channelnum2,3, padding=1), nn.InstanceNorm2d(channelnum2))
        # self.mlayer2_2 = nn.Sequential(nn.Conv2d(channelnum2,channelnum2,3, padding=1), nn.InstanceNorm2d(channelnum2))
        # self.mlayer2_3 = nn.Sequential(nn.Conv2d(channelnum2,channelnum2,3, padding=1), nn.InstanceNorm2d(channelnum2))
        # self.mlayer2_4 = nn.Sequential(nn.Conv2d(channelnum2,channelnum2,3, padding=1), nn.InstanceNorm2d(channelnum2))
        # self.mlayer3 = nn.Sequential(nn.Conv2d(channelnum2,channelnum,3, padding=1), nn.InstanceNorm2d(channelnum))
        # self.mlayer4 = nn.Sequential(nn.Conv2d(channelnum,2,3, padding=1))

        self.blayer1 = nn.Sequential(nn.Conv2d(3,channelnum,3, padding=1))
        self.blayer2 = nn.Sequential(nn.Conv2d(channelnum,channelnum,3, padding=1))
        channelnum2 = 32
        self.blayer2_1 = nn.Sequential(nn.Conv2d(channelnum,channelnum2,3, padding=1))
        self.blayer2_2 = nn.Sequential(nn.Conv2d(channelnum2,channelnum2,3, padding=1))
        self.blayer2_3 = nn.Sequential(nn.Conv2d(channelnum2,channelnum2,3, padding=1))
        self.blayer2_4 = nn.Sequential(nn.Conv2d(channelnum2,channelnum2,3, padding=1))
        self.blayer3 = nn.Sequential(nn.Conv2d(channelnum2,channelnum,3, padding=1))
        self.blayer4 = nn.Sequential(nn.Conv2d(channelnum,3,3, padding=1))

        #  mask block
        self.mlayer1 = nn.Sequential(nn.Conv2d(3,channelnum,3, padding=1))
        self.mlayer2 = nn.Sequential(nn.Conv2d(channelnum,channelnum,3, padding=1))
        self.mlayer2_1 = nn.Sequential(nn.Conv2d(channelnum,channelnum2,3, padding=1))
        self.mlayer2_2 = nn.Sequential(nn.Conv2d(channelnum2,channelnum2,3, padding=1))
        self.mlayer2_3 = nn.Sequential(nn.Conv2d(channelnum2,channelnum2,3, padding=1))
        self.mlayer2_4 = nn.Sequential(nn.Conv2d(channelnum2,channelnum2,3, padding=1))
        self.mlayer3 = nn.Sequential(nn.Conv2d(channelnum2,channelnum,3, padding=1))
        self.mlayer4 = nn.Sequential(nn.Conv2d(channelnum,3,3, padding=1))

    def forward(self, x):
        phi1 = self.blayer1(x)
        phi1r = F.relu(phi1)
        phi2 = self.blayer2(phi1r)
        phi2r = F.relu(phi2)
        bout_lay21 = self.blayer2_1(phi2r)#border out layer 21
        bout_lay21r = F.relu(bout_lay21)
        bout_lay22 = self.blayer2_2(bout_lay21r)
        bout_lay22r = F.relu(bout_lay22)
        bout_lay23 = self.blayer2_3(bout_lay22r)
        bout_lay23r = F.relu(bout_lay23)
        bout_lay24 = self.blayer2_4(bout_lay23r)
        bout_lay24r = F.relu(bout_lay24)
        phi3 = self.blayer3(bout_lay24r)
        phi3r = F.relu(phi3)
        phi4 = self.blayer4(phi3r)
        phi4r = F.relu(phi4)

        psi1 = self.mlayer1(phi4r)
        psi1r = F.relu(psi1)
        psi2 = self.mlayer2(psi1r)
        psi2r = F.relu(psi2)
        mout_lay21 = self.mlayer2_1(psi2r)#mask out layers
        mout_lay21r = F.relu(mout_lay21)
        mout_lay22 = self.mlayer2_2(mout_lay21r)
        mout_lay22r = F.relu(mout_lay22)
        mout_lay23 = self.mlayer2_3(mout_lay22r)
        mout_lay23r = F.relu(mout_lay23)
        mout_lay24 = self.mlayer2_4(mout_lay23r)
        mout_lay24r = F.relu(mout_lay24)
        psi3 = self.mlayer3(mout_lay24r)
        psi3r = F.relu(psi3)
        psi4 = self.mlayer4(psi3r)
        psi4r = F.relu(psi4)
        return phi1, phi2, bout_lay21, bout_lay22, bout_lay23, bout_lay24, phi3, phi4r, psi1, psi2, \
            mout_lay21, mout_lay22, mout_lay23, mout_lay24, psi3, psi4r


class BPNet(nn.Module):
    """net for boundary aware perceptual loss based training
    """
    def __init__(self, in_classes=1, channelscale=64):
        super(BPNet, self).__init__()
        self.unetvgg = UNetSimple(in_classes=in_classes, channelscale=channelscale)#UNetVgg()
        self.cyclenet = CycleNet()
        self.sft = nn.Softmax2d()

    def forward(self, x):
        x_map = self.unetvgg(x)
        x_mapsft = self.sft(x_map)
        cycle_out = self.cyclenet(x_mapsft)
        # ttt=cycle_out[0],cycle_out[1],cycle_out[2],cycle_out[-1],cycle_out[3], cycle_out[4], cycle_out[5], cycle_out[6]
        return x_map, cycle_out