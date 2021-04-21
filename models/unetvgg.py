import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

def make_decoder_block(in_channels, middle_channels, out_channels, if_convtrans=False, noupsam=False, upsc=2):
    """make decoder layer of unet

    Parameters
    ----------
    in_channels : int
        input channel count
    middle_channels :int
        middle channel count
    out_channels : int
        output channel count
    if_convtrans : bool, optional
        if transposed convolution to be used, by default False
    noupsam : bool, optional
        if output array not to be upsampled, by default False

    Returns
    -------
    tensor
        output from decoder layer
    """
    if noupsam:
        upsam = nn.ModuleList([nn.Conv2d(middle_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels)])
    elif if_convtrans:
        upsam = nn.ModuleList([nn.ConvTranspose2d(
                middle_channels, out_channels, kernel_size=4, stride=2, padding=1)])
    else:
        upsam = nn.ModuleList([nn.Conv2d(middle_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.Upsample(scale_factor=upsc, mode='bilinear', align_corners=False)])
    return nn.Sequential(
        nn.Conv2d(in_channels, middle_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        *upsam,
        nn.ReLU(inplace=True))
    # else:
    #     return nn.Sequential(
    #         nn.Conv2d(in_channels, middle_channels, 3, padding=1),
    #         nn.InstanceNorm2d(middle_channels),
    #         nn.ReLU(inplace=True),
    #         nn.Conv2d(middle_channels, out_channels, 3, padding=1),
    #         nn.InstanceNorm2d(out_channels),
    #         nn.Upsample(scale_factor=2, mode='bilinear'),
    #         nn.ReLU(inplace=True))
# In[12]:
class UNetVgg(nn.Module):
# adapted from https://github.com/skorch-dev/skorch/blob/master/examples/nuclei_image_segmentation/model.py
    def __init__(self, pretrained=False, out_classes=3, patch_train=False, if_convtrans=False):
        """unet with encoding layers from vgg

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
        """
        super(UNetVgg, self).__init__()
        self.patch_train = patch_train
        encoder = models.vgg16_bn(pretrained=pretrained).features

        self.conv1 = encoder[:6]
        self.conv2 = encoder[6:13]
        self.conv3 = encoder[13:23]
        self.conv4 = encoder[23:33]
        self.conv5 = encoder[33:43]

        self.center = nn.Sequential(
            encoder[43],  # MaxPool
            make_decoder_block(512, 512, 256, if_convtrans=if_convtrans))

        self.dec5 = make_decoder_block(256 + 512, 512, 256, if_convtrans=if_convtrans)
        self.dec4 = make_decoder_block(256 + 512, 512, 256, if_convtrans=if_convtrans)
        self.dec3 = make_decoder_block(256 + 256, 256, 64, if_convtrans=if_convtrans)
        self.dec2 = make_decoder_block(64 + 128, 128, 32, if_convtrans=if_convtrans)
        self.dec1 = nn.Sequential(
            nn.Conv2d(32 + 64, 32, 3, padding=1), nn.ReLU(inplace=True))
        self.final = nn.Conv2d(32, out_classes, kernel_size=1)

    def forward(self, x):
            conv1 = self.conv1(x)
            conv2 = self.conv2(conv1)
            conv3 = self.conv3(conv2)
            conv4 = self.conv4(conv3)
            conv5 = self.conv5(conv4)

            center = self.center(conv5)

            dec5 = self.dec5(torch.cat([center, conv5], 1))
            dec4 = self.dec4(torch.cat([dec5, conv4], 1))
            dec3 = self.dec3(torch.cat([dec4, conv3], 1))
            dec2 = self.dec2(torch.cat([dec3, conv2], 1))
            dec1 = self.dec1(torch.cat([dec2, conv1], 1))

            return self.final(dec1)

class UNetVgg256(nn.Module):
    # adapted from https://github.com/skorch-dev/skorch/blob/master/examples/nuclei_image_segmentation/model.py
    """
    intended to be used with patch aggregation based training
    reshape input to 128, pass it to unetvgg and gets outputs
    TODO:then reshape output to original
    """
    def __init__(self, out_classes=3):
        """unet with encoding layers from vgg

        Parameters
        ----------
        out_classes : int, optional
            count of channels in output tensor, by default 3
        """
        super(UNetVgg256, self).__init__()
        self.unetvgg = UNetVgg()

    def forward(self, x):
        orig_shape = x.shape[-2:]
        inter_x = F.interpolate(x, [128, 128])
        out = self.unetvgg.forward(inter_x)
        last = torch.nn.AdaptiveMaxPool2d(orig_shape)
        # out = F.interpolate(out, orig_shape, mode='bicubic', align_corners=True)
        out=last(out)
        return out

class UNetVggWholeImg(nn.Module):
    # adapted from https://github.com/skorch-dev/skorch/blob/master/examples/nuclei_image_segmentation/model.py
    """
    reshape input to 256, pass it to unetvgg and then reshape to original
    """
    def __init__(self, save_model_path, out_classes=3):
        """
        Parameters
        ----------
        save_model_path : path like
            torch model saved path
        out_classes : int, optional
            number of output channels, by default 3
        """
        super(UNetVggWholeImg, self).__init__()
        self.unetvgg = UNetVgg()
        checkpoint = torch.load(save_model_path)
        self.unetvgg.load_state_dict(checkpoint['state_dict'])

    def forward(self, x):
        out = self.unetvgg.forward(x)
        return out

# TODO replace UNetVgg with below class
class UNetVggInOut(nn.Module):
    """allows varying input channels
    """
# adapted from https://github.com/skorch-dev/skorch/blob/master/examples/nuclei_image_segmentation/model.py
    def __init__(self, pretrained=False, in_classes=1, out_classes=3, patch_train=False):
        """
        Parameters
        ----------
        pretrained : bool, optional
            [description], by default False
        in_classes : int, optional
            [description], by default 1
        out_classes : int, optional
            [description], by default 3
        patch_train : bool, optional
            [description], by default False
        """
        super(UNetVggInOut, self).__init__()
        self.patch_train = patch_train
        encoder = models.vgg16_bn(pretrained=pretrained).features
        self.conv0 = nn.Conv2d(in_classes, 3, 3, padding=1)
        self.conv1 = encoder[:6]
        self.conv2 = encoder[6:13]
        self.conv3 = encoder[13:23]
        self.conv4 = encoder[23:33]
        self.conv5 = encoder[33:43]

        self.center = nn.Sequential(
            encoder[43],  # MaxPool
            make_decoder_block(512, 512, 256))

        self.dec5 = make_decoder_block(256 + 512, 512, 256)
        self.dec4 = make_decoder_block(256 + 512, 512, 256)
        self.dec3 = make_decoder_block(256 + 256, 256, 64)
        self.dec2 = make_decoder_block(64 + 128, 128, 32)
        self.dec1 = nn.Sequential(
            nn.Conv2d(32 + 64, 32, 3, padding=1), nn.ReLU(inplace=True))
        self.final = nn.Conv2d(32, out_classes, kernel_size=1)

    def forward(self, x):
            conv0 = self.conv0(x)
            conv1 = self.conv1(conv0)
            conv2 = self.conv2(conv1)
            conv3 = self.conv3(conv2)
            conv4 = self.conv4(conv3)
            conv5 = self.conv5(conv4)

            center = self.center(conv5)

            dec5 = self.dec5(torch.cat([center, conv5], 1))
            dec4 = self.dec4(torch.cat([dec5, conv4], 1))
            dec3 = self.dec3(torch.cat([dec4, conv3], 1))
            dec2 = self.dec2(torch.cat([dec3, conv2], 1))
            dec1 = self.dec1(torch.cat([dec2, conv1], 1))

            return self.final(dec1)


class UNetVggTwo(nn.Module):
    # adapted from https://github.com/skorch-dev/skorch/blob/master/examples/nuclei_image_segmentation/model.py
    """
    two networks one for left half and other for right half of input(ususally rectangular image)
    expecting a rectangular input image cropped close enough to region of segmentation
    """
    def __init__(self):
        super(UNetVggTwo, self).__init__()
        # for left half
        self.unetvggl = UNetVggInOut()
        self.unetvggr = UNetVggInOut()

    def forward(self, x, y=None):
        # x into 2 parts, if width count odd, righthalf with one more pixel
        if y is not None:
            # finding width (min,max), height where fg labels lie         
            hmin, hmax = torch.nonzero(y, as_tuple=True)[1].min(), torch.nonzero(y, as_tuple=True)[1].max()
            wmin, wmax = torch.nonzero(y, as_tuple=True)[2].min(), torch.nonzero(y, as_tuple=True)[2].max()
            margin=15
            orig_hmax = y.shape[-2]
            orig_wmax = y.shape[-1]
            crop_hmin, crop_hmax = max(0, hmin-margin), min(hmax+margin, orig_hmax)
            crop_wmin, crop_wmax = max(0, wmin-margin), min(wmax+margin, orig_wmax)
            x = x[:,:, crop_hmin: crop_hmax, crop_wmin: crop_wmax].clone()
            # y_crop = y[:,:,max(0, hmin-margin): min(hmax+margin, orig_hmax), max(0, wmin-margin): min(wmax+margin, orig_wmax)].clone()

        width = x.shape[-1]
        height = x.shape[-2]
        xl = x[:,:,:,:int(width/2)].clone()
        xr = x[:,:,:,int(width/2):].clone()

        inter_xl = F.interpolate(xl, [256, 256])
        inter_xr = F.interpolate(xr, [256, 256])
        outl = self.unetvggl.forward(inter_xl)
        outr = self.unetvggr.forward(inter_xr)
        # now outl and outr of 128X128, and needed to be combined them to original shape
        lastl = torch.nn.AdaptiveMaxPool2d([height,int(width/2)] )
        lastr = torch.nn.AdaptiveMaxPool2d([height,width-int(width/2)] )
        outl=lastl(outl)
        outr=lastr(outr)
        # outl = F.interpolate(outl, [height,int(width/2)] , mode='bicubic', align_corners=True)
        # outr = F.interpolate(outr, [height,width-int(width/2)], mode='bicubic', align_corners=True)

        # concatenating left and right parts
        out = torch.cat((outl, outr), dim=-1)
        if y is not None:
            # means x had been cropped inside model, so restore to original dim
            # pad last dim by (crop_wmin, crop_wmax) and 2nd to last by (crop_hmin-1, orig_hmax-crop_hmax+1)
            out = F.pad(out, (crop_wmin-1, orig_wmax-crop_wmax+1, crop_hmin-1, orig_hmax-crop_hmax+1))
        return out


def make_encoder_block(in_channels, middle_channels, out_channels, convs3=True, middle_channels2=None, max_pool=False,
no_dsam=False, if_interpolate=False, downsc=2):
    """contraction unit

    Parameters
    ----------
    in_channels : int
        input channels
    middle_channels : int
        middle channels
    out_channels : int
        output channels
    convs3 : bool, optional
        whether to include 2 conv2d, by default True
    middle_channels2 : int, optional
        second middle channel, by default None
    max_pool : bool, optional
        whether max pool or transposed conv, by default False
    no_dsam : bool, optional
        whether to downsample, by default False
    if_interpolate : bool, optional
        whether to interpolate, by default False
    downsc : int, optional
        whether to downscale, by default 2

    Returns
    -------
    `torch.nn. Sequential`
        encoding layers
    """
    if if_interpolate:
        down_func = nn.Upsample(scale_factor=downsc, mode="bilinear", align_corners=False)
    elif no_dsam:
        down_func = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
    elif max_pool:
        down_func = nn.MaxPool2d(kernel_size=2, stride=2)
    else:
        down_func = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2)
    if convs3:
        mid_layer = nn.ModuleList([nn.Conv2d(middle_channels, middle_channels2, 3, padding=1, bias=False),
            nn.InstanceNorm2d(middle_channels2),
            nn.ReLU(inplace=True)])
        last_in = middle_channels2
    else:
        mid_layer=[]
        last_in = middle_channels
    return nn.Sequential(
        down_func,
        nn.Conv2d(in_channels, middle_channels, 3, padding=1, bias=False),
        nn.InstanceNorm2d(middle_channels),
        nn.ReLU(inplace=True),
        *mid_layer,
        nn.Conv2d(last_in, out_channels, 3, padding=1, bias=False),
        nn.InstanceNorm2d(out_channels),
        nn.ReLU(inplace=True)
        )
    # else:
    #     return nn.Sequential(
    #         down_func,
    #         nn.Conv2d(in_channels, middle_channels, 3, padding=1, bias=False),
    #         nn.InstanceNorm2d(middle_channels),
    #         nn.ReLU(inplace=True),
    #         nn.Conv2d(middle_channels, out_channels, 3, padding=1, bias=False),
    #         nn.InstanceNorm2d(out_channels),
    #         nn.ReLU(inplace=True)
    #     )

def make_center(in_channels, max_pool=True, iftrans2d=False):
    """make center unit

    Parameters
    ----------
    in_channels : int
        input channel count
    max_pool : bool, optional
        whether max pool, by default True
    iftrans2d : bool, optional
        is transposed conv2d, by default False

    Returns
    -------
    `torch.nn. Sequential`
        sequential layers
    """
    if max_pool:
        down_func = nn.MaxPool2d(kernel_size=2, stride=2)
    else:
        down_func = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2)
    if iftrans2d:
        return nn.Sequential(
                down_func,
                nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1,1)),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=1),
                nn.ReLU(inplace=True)
                )
    else:
        return nn.Sequential(
                down_func,
                nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1,1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 256, 3, padding=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.ReLU(inplace=True)
                )
class UNetNaive(nn.Module):
# adapted from https://github.com/skorch-dev/skorch/blob/master/examples/nuclei_image_segmentation/model.py
    def __init__(self, pretrained=False, out_classes=3, patch_train=False, in_classes=1, if_convtrans=False, ifmaxpool=False):
            """

        Parameters
        ----------
        pretrained : bool
            use previously trained model
        out_classes : int
            input channels
        patch_train : bool
            whether for patch train
        in_classes : int
            input channels
        if_convtrans : bool, optional
            whether to use transposed conv2d
        ifmaxpool : bool, optional
            whether max pool or transposed conv, by default False
            """
            super(UNetNaive, self).__init__()
            self.patch_train = patch_train
            self.conv1 = nn.Sequential(
            nn.Conv2d(in_classes, 64, 3, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
            )
            self.conv2 = make_encoder_block(in_channels=64, middle_channels=128, out_channels=128, convs3=False)
            self.conv3 = make_encoder_block(in_channels=128, middle_channels=256, out_channels=256, middle_channels2=256)
            self.conv4 = make_encoder_block(in_channels=256, middle_channels=512, out_channels=512, middle_channels2=512)
            self.conv5 = make_encoder_block(in_channels=512, middle_channels=512, out_channels=512, middle_channels2=512)

            self.center = make_center(in_channels=512, max_pool=ifmaxpool, iftrans2d=if_convtrans)
            # nn.Sequential(
            #     nn.MaxPool2d(kernel_size=2, stride=2),
            #     nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1,1)),
            #     nn.ReLU(inplace=True),
            #     nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=1),
            #     nn.ReLU(inplace=True)
            #     )

            self.dec5 = make_decoder_block(256 + 512, 512, 256)
            self.dec4 = make_decoder_block(256 + 512, 512, 256)
            self.dec3 = make_decoder_block(256 + 256, 256, 64)
            self.dec2 = make_decoder_block(64 + 128, 128, 32)
            self.dec1 = nn.Sequential(
                nn.Conv2d(32 + 64, 32, 3, padding=1), nn.ReLU(inplace=True))
            self.final = nn.Conv2d(32, out_classes, kernel_size=1)

    def forward(self, x):
            conv1 = self.conv1(x)
            conv2 = self.conv2(conv1)
            conv3 = self.conv3(conv2)
            conv4 = self.conv4(conv3)
            conv5 = self.conv5(conv4)

            center = self.center(conv5)

            dec5 = self.dec5(torch.cat([center, conv5], 1))
            dec4 = self.dec4(torch.cat([dec5, conv4], 1))
            dec3 = self.dec3(torch.cat([dec4, conv3], 1))
            dec2 = self.dec2(torch.cat([dec3, conv2], 1))
            dec1 = self.dec1(torch.cat([dec2, conv1], 1))

            return self.final(dec1)


class UNetWithoutCenter(nn.Module):
    def __init__(self, pretrained=False, out_classes=3, patch_train=False, in_classes=1):
            """
        without central bottleneck
        Parameters
        ----------
        pretrained : bool
            use previously trained model
        out_classes : int
            input channels
        patch_train : bool
            whether for patch train
        in_classes : int
            input channels
            """
            super(UNetWithoutCenter, self).__init__()
            self.patch_train = patch_train
            self.conv1 = nn.Sequential(
            nn.Conv2d(in_classes, 64, 3, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
            )
            self.conv2 = make_encoder_block(in_channels=64, middle_channels=128, out_channels=128, convs3=False)
            self.conv3 = make_encoder_block(in_channels=128, middle_channels=256, out_channels=256, middle_channels2=256)
            self.conv4 = make_encoder_block(in_channels=256, middle_channels=512, out_channels=512, middle_channels2=512)
            self.conv5 = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                nn.InstanceNorm2d(512),
                nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1,1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
                nn.InstanceNorm2d(256),
                # nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.ReLU(inplace=True)
                )

            # self.dec5 = make_decoder_block(512, 512, 256)
            self.dec4 = make_decoder_block(256 + 512, 512, 256)
            self.dec3 = make_decoder_block(256 + 256, 256, 64)
            self.dec2 = make_decoder_block(64 + 128, 128, 32)
            self.dec1 = nn.Sequential(
                nn.Conv2d(32 + 64, 32, 3, padding=1), nn.ReLU(inplace=True))
            self.final = nn.Conv2d(32, out_classes, kernel_size=1)

    def forward(self, x):
            # s(x)[:,0,:,:].unsqueeze(dim=1).expand([7,32,256,256])
            conv1 = self.conv1(x)
            conv2 = self.conv2(conv1)
            conv3 = self.conv3(conv2)
            conv4 = self.conv4(conv3)
            conv5 = self.conv5(conv4)

            # center = self.center(conv5)

            # dec5 = self.dec5(conv5)
            dec4 = self.dec4(torch.cat([conv5, conv4], 1))
            dec3 = self.dec3(torch.cat([dec4, conv3], 1))
            dec2 = self.dec2(torch.cat([dec3, conv2], 1))
            dec1 = self.dec1(torch.cat([dec2, conv1], 1))

            return self.final(dec1)


class UNetNaiveMultiOut(nn.Module):
    """unet naive implementation with multiple outputs
    """
# adapted from https://github.com/skorch-dev/skorch/blob/master/examples/nuclei_image_segmentation/model.py
    def __init__(self, pretrained=False, out_classes=3, patch_train=False, in_classes=1):
            """
        Parameters
        ----------
        pretrained : bool
            use previously trained model
        out_classes : int
            input channels
        patch_train : bool
            whether for patch train
        in_classes : int
            input channels
            """
            super(UNetNaiveMultiOut, self).__init__()
            self.patch_train = patch_train
            self.conv1 = nn.Sequential(
            nn.Conv2d(in_classes, 64, 3, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
            )
            self.conv2 = make_encoder_block(in_channels=64, middle_channels=128, out_channels=128, convs3=False)
            self.conv3 = make_encoder_block(in_channels=128, middle_channels=256, out_channels=256, middle_channels2=256)
            self.conv4 = make_encoder_block(in_channels=256, middle_channels=512, out_channels=512, middle_channels2=512)
            self.conv5 = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(512),
                # nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1,1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(256),
                # nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.ReLU(inplace=True)
                )

            # self.dec5 = make_decoder_block(512, 512, 256)
            self.dec4 = make_decoder_block(256 + 512, 512, 256)
            self.dec3 = make_decoder_block(256 + 256, 256, 64)
            self.dec2 = make_decoder_block(64 + 128, 128, 32)
            self.dec1 = nn.Sequential(
                nn.Conv2d(32 + 64, 32, 3, padding=1), nn.ReLU(inplace=True))
            self.final = nn.Conv2d(32, out_classes, kernel_size=1)

    def forward(self, x):
            # s(x)[:,0,:,:].unsqueeze(dim=1).expand([7,32,256,256])
            conv1 = self.conv1(x)
            conv2 = self.conv2(conv1)
            conv3 = self.conv3(conv2)
            conv4 = self.conv4(conv3)
            conv5 = self.conv5(conv4)

            # center = self.center(conv5)

            # dec5 = self.dec5(conv5)
            dec4 = self.dec4(torch.cat([conv5, conv4], 1))
            dec3 = self.dec3(torch.cat([dec4, conv3], 1))
            dec2 = self.dec2(torch.cat([dec3, conv2], 1))
            dec1 = self.dec1(torch.cat([dec2, conv1], 1))

            return self.final(dec1), [conv3, dec4]

class UNetLimMultiOut(nn.Module):
    """unet naive implementation with multiple outputs, max channle number per layer=16
    """
# adapted from https://github.com/skorch-dev/skorch/blob/master/examples/nuclei_image_segmentation/model.py
    def __init__(self, pretrained=False, out_classes=3, patch_train=False, in_classes=1, channel_depth=16, multiout=False):
            """
        Parameters
        ----------
        pretrained : bool
            use previously trained model
        out_classes : int
            input channels
        patch_train : bool
            whether for patch train
        in_classes : int
            input channels
        channel_depth : int
            channel factor
        multiout : bool
            whether multiple outputs
            """
            super(UNetLimMultiOut, self).__init__()
            self.multiout=multiout
            self.patch_train = patch_train
            self.conv1 = nn.Sequential(
            nn.Conv2d(in_classes, channel_depth, 3, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_depth, channel_depth, 3, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
            )
            self.conv2 = make_encoder_block(in_channels=channel_depth, middle_channels=channel_depth, out_channels=channel_depth, convs3=False)
            self.conv3 = make_encoder_block(in_channels=channel_depth, middle_channels=channel_depth, out_channels=channel_depth, middle_channels2=channel_depth)
            self.conv4 = make_encoder_block(in_channels=channel_depth, middle_channels=channel_depth, out_channels=channel_depth, middle_channels2=channel_depth)
            self.conv5 = nn.Sequential(
                nn.Conv2d(in_channels=channel_depth, out_channels=channel_depth, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(channel_depth),
                # nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1,1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=channel_depth, out_channels=channel_depth//2, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(channel_depth),
                # nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.ReLU(inplace=True)
                )

            # self.dec5 = make_decoder_block(512, 512, 256)
            self.dec4 = make_decoder_block(channel_depth+channel_depth//2, channel_depth, channel_depth//2)
            self.dec3 = make_decoder_block(channel_depth+channel_depth//2, channel_depth, channel_depth//2)
            self.dec2 = make_decoder_block(channel_depth+channel_depth//2, channel_depth, channel_depth//2)
            self.dec1 = nn.Sequential(
                nn.Conv2d(channel_depth+channel_depth//2, channel_depth, 3, padding=1), nn.ReLU(inplace=True))
            self.final = nn.Conv2d(channel_depth, out_classes, kernel_size=1)

    def forward(self, x):
            # s(x)[:,0,:,:].unsqueeze(dim=1).expand([7,32,256,256])
            conv1 = self.conv1(x)
            conv2 = self.conv2(conv1)
            conv3 = self.conv3(conv2)
            conv4 = self.conv4(conv3)
            conv5 = self.conv5(conv4)

            # center = self.center(conv5)

            # dec5 = self.dec5(conv5)
            dec4 = self.dec4(torch.cat([conv5, conv4], 1))
            dec3 = self.dec3(torch.cat([dec4, conv3], 1))
            dec2 = self.dec2(torch.cat([dec3, conv2], 1))
            dec1 = self.dec1(torch.cat([dec2, conv1], 1))
            if self.multiout:
                return self.final(dec1), conv3
            else:
                return self.final(dec1)


class UNetSimple(nn.Module):
# adapted from https://github.com/skorch-dev/skorch/blob/master/examples/nuclei_image_segmentation/model.py
    def __init__(self, pretrained=False, out_classes=3, patch_train=False, in_classes=1, channelscale=64, 
    multiout=False, noupdownsam=False):
            """
            Parameters
            ----------
            pretrained : bool
                use previously trained model
            out_classes : int
                input channels
            patch_train : bool
                whether for patch train
            in_classes : int
                input channels
            channel_scale : int
                channel factor
            multiout : bool
                whether multiple outputs
            noupdownsam : bool
                whether to vary scale
            """
            super(UNetSimple, self).__init__()
            self.multiout=multiout
            self.patch_train = patch_train
            self.conv1 = nn.Sequential(
            nn.Conv2d(in_classes, channelscale, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channelscale),
            nn.ReLU(inplace=True),
            nn.Conv2d(channelscale, channelscale, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channelscale),
            nn.ReLU(inplace=True)
            )
            self.conv2 = make_encoder_block(in_channels=channelscale, middle_channels=2*channelscale, 
            out_channels=2*channelscale, convs3=False, no_dsam=noupdownsam)
            self.conv3 = make_encoder_block(in_channels=2*channelscale, middle_channels=4*channelscale, 
            out_channels=4*channelscale, middle_channels2=4*channelscale, no_dsam=noupdownsam)
            self.conv4 = make_encoder_block(in_channels=4*channelscale, middle_channels=8*channelscale, 
            out_channels=8*channelscale, middle_channels2=8*channelscale, no_dsam=noupdownsam)
            self.conv5 = nn.Sequential(
                nn.Conv2d(in_channels=8*channelscale, out_channels=8*channelscale, kernel_size=3, padding=1),
                nn.InstanceNorm2d(512),
                nn.Conv2d(8*channelscale, 8*channelscale, kernel_size=(3, 3), padding=(1,1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=8*channelscale, out_channels=8*channelscale, kernel_size=3, padding=1),
                nn.InstanceNorm2d(4*channelscale),
                # nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.ReLU(inplace=True)
                )

            # self.dec5 = make_decoder_block(512, 512, 256)
            self.dec4 = make_decoder_block(16*channelscale, 8*channelscale, 4*channelscale, noupsam=noupdownsam)
            self.dec3 = make_decoder_block(8*channelscale, 4*channelscale, channelscale, noupsam=noupdownsam)
            self.dec2 = make_decoder_block(3*channelscale, 2*channelscale, channelscale//2, noupsam=noupdownsam)
            self.dec1 = nn.Sequential(
                nn.Conv2d(int(1.5*channelscale), channelscale//2, 3, padding=1), nn.ReLU(inplace=True))
            self.final = nn.Conv2d(channelscale//2, out_classes, kernel_size=1)

    def forward(self, x):
            # s(x)[:,0,:,:].unsqueeze(dim=1).expand([7,32,256,256])
            conv1 = self.conv1(x)
            conv2 = self.conv2(conv1)
            conv3 = self.conv3(conv2)
            conv4 = self.conv4(conv3)
            conv5 = self.conv5(conv4)

            # center = self.center(conv5)

            # dec5 = self.dec5(conv5)
            dec4 = self.dec4(torch.cat([conv5, conv4], 1))
            dec3 = self.dec3(torch.cat([dec4, conv3], 1))
            dec2 = self.dec2(torch.cat([dec3, conv2], 1))
            dec1 = self.dec1(torch.cat([dec2, conv1], 1))

            if self.multiout:
                return self.final(dec1), [dec1,dec2, dec3, dec4]#[conv4, conv5]
            else:
                return self.final(dec1)

class UNetSkip(nn.Module):
# adapted from https://github.com/skorch-dev/skorch/blob/master/examples/nuclei_image_segmentation/model.py
    def __init__(self, pretrained=False, out_classes=3, patch_train=False, in_classes=1, channelscale=32, 
    multiout=False, noupdownsam=False):
            """include more connections

            Parameters
            ----------
            pretrained : bool, optional
                use previously trained model, by default False
            out_classes : int, optional
                output classes, by default 3
            patch_train : bool, optional
                whether patch training, by default False
            in_classes : int, optional
                input channels, by default 1
            channelscale : int, optional
                channel factor, by default 32
            multiout : bool, optional
                whether return multiple outputs, by default False
            noupdownsam : bool, optional
                whether to vary image scale, by default False
            """
            super(UNetSkip, self).__init__()
            self.xcontr = 3
            self.multiout=multiout
            self.patch_train = patch_train
            self.conv1 = nn.Sequential(
            nn.Conv2d(in_classes, channelscale, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channelscale),
            nn.ReLU(inplace=True),
            nn.Conv2d(channelscale, channelscale, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channelscale),
            nn.ReLU(inplace=True)
            )
            self.conv2 = make_encoder_block(in_channels=channelscale, middle_channels=2*channelscale, 
            out_channels=2*channelscale, convs3=False, no_dsam=noupdownsam)
            self.conv3 = make_encoder_block(in_channels=2*channelscale, middle_channels=4*channelscale, 
            out_channels=4*channelscale, middle_channels2=4*channelscale, no_dsam=noupdownsam)
            self.conv4 = make_encoder_block(in_channels=4*channelscale, middle_channels=8*channelscale, 
            out_channels=8*channelscale, middle_channels2=8*channelscale, no_dsam=noupdownsam)
            self.conv5 = nn.Sequential(
                nn.Conv2d(in_channels=8*channelscale, out_channels=8*channelscale, kernel_size=3, padding=1),
                nn.InstanceNorm2d(512),
                nn.Conv2d(8*channelscale, 8*channelscale, kernel_size=(3, 3), padding=(1,1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=8*channelscale, out_channels=8*channelscale, kernel_size=3, padding=1),
                nn.InstanceNorm2d(4*channelscale),
                # nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.ReLU(inplace=True)
                )

            # self.dec5 = make_decoder_block(512, 512, 256)
            self.dec4 = make_decoder_block(16*channelscale, 8*channelscale, 4*channelscale, noupsam=noupdownsam)
            self.dec3 = make_decoder_block(8*channelscale, 4*channelscale, channelscale, noupsam=noupdownsam)
            self.dec2 = make_decoder_block(3*channelscale, 2*channelscale, channelscale//2, noupsam=noupdownsam)
            self.dec1 = nn.Sequential(
                nn.Conv2d(int(1.5*channelscale), channelscale//2, 3, padding=1), nn.ReLU(inplace=True))
            self.final = nn.Conv2d(channelscale//2, out_classes, kernel_size=1)

    def forward(self, x):
            # s(x)[:,0,:,:].unsqueeze(dim=1).expand([7,32,256,256])
            conv1 = self.conv1(x)
            conv2 = self.conv2(conv1)
            #contribution from x into conv3
            xconv3 = F.upsample(x[...,:self.xcontr], [conv2.shape[-2], conv2.shape[-1]], mode='bilinear', align_corners=False)
            conv3 = self.conv3(conv2)
            conv4 = self.conv4(conv3)
            conv5 = self.conv5(conv4)

            # center = self.center(conv5)

            # dec5 = self.dec5(conv5)
            dec4 = self.dec4(torch.cat([conv5, conv4], 1))
            xdec3 = F.upsample(x[...,:self.xcontr], [dec4.shape[-2], dec4.shape[-1]], mode='bilinear', align_corners=False)
            dec3 = self.dec3(torch.cat([dec4, conv3], 1))
            dec2 = self.dec2(torch.cat([dec3, conv2], 1))
            xdec1 = F.upsample(x[...,:self.xcontr], [dec2.shape[-2], dec2.shape[-1]], mode='bilinear', align_corners=False)
            dec1 = self.dec1(torch.cat([dec2, conv1], 1))

            if self.multiout:
                return self.final(dec1), dec4#[conv4, conv5]
            else:
                return self.final(dec1)

class UNetSimpleReduced(nn.Module):
# adapted from https://github.com/skorch-dev/skorch/blob/master/examples/nuclei_image_segmentation/model.py
    """unetsimple model reduced by one more layer
    """
    def __init__(self, pretrained=False, out_classes=3, patch_train=False, in_classes=1, channelscale=64, 
    multiout=False, noupdownsam=False):
        """
        Parameters
        ----------
        pretrained : bool
            use previously trained model
        out_classes : int, optional
            output channel count, by default 3
        patch_train : bool, optional
            if patches are trained, by default False
        in_classes : int, optional
            input classes, by default 1
        channelscale : int, optional
            channel scaling factor, by default 64
        multiout : bool, optional
            if mutliple outputs are returned, by default False
        noupdownsam : bool, optional
            if up/down sampling applied, by default False
        """
        super(UNetSimpleReduced, self).__init__()
        self.multiout=multiout
        self.patch_train = patch_train
        self.conv1 = nn.Sequential(
        nn.Conv2d(in_classes, channelscale, 3, padding=1, bias=False),
        nn.InstanceNorm2d(channelscale),
        nn.ReLU(inplace=True),
        nn.Conv2d(channelscale, channelscale, 3, padding=1, bias=False),
        nn.InstanceNorm2d(channelscale),
        nn.ReLU(inplace=True)
        )
        self.conv2 = make_encoder_block(in_channels=channelscale, middle_channels=2*channelscale, 
        out_channels=2*channelscale, convs3=False, no_dsam=noupdownsam)
        self.conv3 = make_encoder_block(in_channels=2*channelscale, middle_channels=4*channelscale, 
        out_channels=8*channelscale, middle_channels2=4*channelscale, no_dsam=noupdownsam)
        # self.conv4 = make_encoder_block(in_channels=4*channelscale, middle_channels=8*channelscale, 
        # out_channels=8*channelscale, middle_channels2=8*channelscale, no_dsam=noupdownsam)
        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(in_channels=8*channelscale, out_channels=8*channelscale, kernel_size=3, padding=1),
        #     nn.InstanceNorm2d(512),
        #     nn.Conv2d(8*channelscale, 8*channelscale, kernel_size=(3, 3), padding=(1,1)),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=8*channelscale, out_channels=8*channelscale, kernel_size=3, padding=1),
        #     nn.InstanceNorm2d(4*channelscale),
        #     # nn.Upsample(scale_factor=2, mode='bilinear'),
        #     nn.ReLU(inplace=True)
        #     )

        # self.dec5 = make_decoder_block(512, 512, 256)
        self.dec4 = make_decoder_block(8*channelscale, 8*channelscale, 4*channelscale, noupsam=True)
        self.dec3 = make_decoder_block(12*channelscale, 4*channelscale, channelscale, noupsam=noupdownsam)
        self.dec2 = make_decoder_block(3*channelscale, 2*channelscale, channelscale//2, noupsam=noupdownsam)
        self.dec1 = nn.Sequential(
            nn.Conv2d(int(1.5*channelscale), channelscale//2, 3, padding=1), nn.ReLU(inplace=True))
        self.final = nn.Conv2d(channelscale//2, out_classes, kernel_size=1)

    def forward(self, x):
            # s(x)[:,0,:,:].unsqueeze(dim=1).expand([7,32,256,256])
            conv1 = self.conv1(x)
            conv2 = self.conv2(conv1)
            conv3 = self.conv3(conv2)
            dec4 = self.dec4(conv3)
            # conv5 = self.conv5(conv4)

            # center = self.center(conv5)

            # dec5 = self.dec5(conv5)
            # dec4 = self.dec4(torch.cat([conv5, conv4], 1))
            dec3 = self.dec3(torch.cat([dec4, conv3], 1))
            dec2 = self.dec2(torch.cat([dec3, conv2], 1))
            dec1 = self.dec1(torch.cat([dec2, conv1], 1))

            if self.multiout:
                return self.final(dec1), [conv4, conv5]
            else:
                return self.final(dec1)

class SpiderNet(nn.Module):
# adapted from https://github.com/skorch-dev/skorch/blob/master/examples/nuclei_image_segmentation/model.py
    """model with different resolution images given as input and then concatenated after up/down sampling to same resolution
    """
    def __init__(self, out_classes=3, patch_train=False, in_classes=1, channelscale=64, 
    multiout=False, noupdownsam=False):
        """
        Parameters
        ----------
        out_classes : int, optional
            output channel count, by default 3
        patch_train : bool, optional
            if patches are trained, by default False
        in_classes : int, optional
            input classes, by default 1
        channelscale : int, optional
            channel scaling factor, by default 64
        multiout : bool, optional
            if mutliple outputs are returned, by default False
        noupdownsam : bool, optional
            if up/down sampling applied, by default False
        """
        super(SpiderNet, self).__init__()
        self.multiout=multiout
        self.patch_train = patch_train

        #conv0 will magnified version of image
        # self.conv_1 = make_encoder_block(in_channels=in_classes, middle_channels=channelscale//4, 
        # out_channels=channelscale//4, convs3=False, no_dsam=noupdownsam, if_interpolate=True, downsc=4)
        self.conv0 = make_encoder_block(in_channels=in_classes, middle_channels=channelscale//2, 
        out_channels=channelscale//2, convs3=False, no_dsam=noupdownsam, if_interpolate=True, downsc=2)

        self.conv1 = nn.Sequential(
        nn.Conv2d(in_classes, channelscale, 3, padding=1, bias=False),
        nn.InstanceNorm2d(channelscale),
        nn.ReLU(inplace=True),
        nn.Conv2d(channelscale, channelscale, 3, padding=1, bias=False),
        nn.InstanceNorm2d(channelscale),
        nn.ReLU(inplace=True)
        )
        self.conv2 = make_encoder_block(in_channels=in_classes, middle_channels=2*channelscale, 
        out_channels=2*channelscale, convs3=False, no_dsam=noupdownsam, if_interpolate=True, downsc=0.5)
        self.conv3 = make_encoder_block(in_channels=in_classes, middle_channels=4*channelscale, 
        out_channels=4*channelscale, middle_channels2=4*channelscale, no_dsam=noupdownsam, if_interpolate=True, downsc=0.25)
        self.conv4 = make_encoder_block(in_channels=in_classes, middle_channels=8*channelscale, 
        out_channels=8*channelscale, middle_channels2=8*channelscale, no_dsam=noupdownsam, if_interpolate=True, downsc=1/8)
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=channelscale+2*channelscale+4*channelscale+8*channelscale, 
            out_channels=8*channelscale, kernel_size=3, padding=1),
            nn.InstanceNorm2d(512),
            nn.Conv2d(12*channelscale, 8*channelscale, kernel_size=(3, 3), padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8*channelscale, out_channels=8*channelscale, kernel_size=3, padding=1),
            nn.InstanceNorm2d(8*channelscale),
            # nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReLU(inplace=True)
            )

        # self.dec5 = make_decoder_block(512, 512, 256)
        self.dec4 = make_decoder_block(8*channelscale, 8*channelscale, 4*channelscale, noupsam=noupdownsam, upsc=8)
        self.dec3 = make_decoder_block(4*channelscale, 4*channelscale, channelscale, noupsam=noupdownsam, upsc=4)
        self.dec2 = make_decoder_block(2*channelscale, 2*channelscale, channelscale//2, noupsam=noupdownsam, upsc=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(channelscale, channelscale//2, 3, padding=1), nn.ReLU(inplace=True))
        self.dec0 = make_decoder_block(channelscale//2, channelscale//2, channelscale//4, noupsam=noupdownsam, upsc=0.5)
        # self.dec_1 = make_decoder_block(channelscale//4, channelscale//4, channelscale//16, noupsam=noupdownsam, upsc=1/4)
        self.final = nn.Conv2d(
        4*channelscale+
        channelscale+
        channelscale//2
        +channelscale//2+channelscale//4,#+channelscale//16, 
        out_classes, kernel_size=1)

    def forward(self, x):
            # s(x)[:,0,:,:].unsqueeze(dim=1).expand([7,32,256,256])
            # conv_1 = self.conv_1(x)
            conv0 = self.conv0(x)
            conv1 = self.conv1(x)
            conv2 = self.conv2(x)
            conv3 = self.conv3(x)
            conv4 = self.conv4(x)
            # conv5 = self.conv5(torch.cat([conv4, conv3, conv2, conv1], 1))

            # center = self.center(conv5)

            # dec5 = self.dec5(conv5)
            dec4 = self.dec4(conv4)
            dec3 = self.dec3(conv3)
            dec2 = self.dec2(conv2)
            dec1 = self.dec1(conv1)
            dec0 = self.dec0(conv0)
            # dec_1 = self.dec_1(conv_1)

            if self.multiout:
                # return self.final(dec1), dec1
                return self.final(torch.cat([dec4, dec3, dec2, dec1, dec0], 1)), [dec0, dec1, dec2,dec3, dec4]#[conv4, conv5]
            else:
                return self.final(torch.cat([dec4, dec3, dec2, dec1, dec0], 1))


class UNetTetra(nn.Module):
# adapted from https://github.com/skorch-dev/skorch/blob/master/examples/nuclei_image_segmentation/model.py
    def __init__(self, pretrained=False, out_classes=3, patch_train=False, in_classes=1, channelscale=64, 
    multiout=False, noupdownsam=False):
            """
            Parameters
            ----------
            pretrained : bool
                use previously trained model
            out_classes : int
                input channels
            patch_train : bool
                whether for patch train
            in_classes : int
                input channels
            channel_scale : int
                channel factor
            multiout : bool
                whether multiple outputs
            noupdownsam : bool
                whether to vary scale
            """
            super(UNetTetra, self).__init__()
            self.multiout=multiout
            self.patch_train = patch_train
            self.conv1 = nn.Sequential(
            nn.Conv2d(in_classes, channelscale, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channelscale),
            nn.ReLU(inplace=True),
            nn.Conv2d(channelscale, channelscale, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channelscale),
            nn.ReLU(inplace=True)
            )
            self.conv2 = make_encoder_block(in_channels=channelscale, middle_channels=2*channelscale, 
            out_channels=2*channelscale, convs3=False, no_dsam=noupdownsam)
            self.conv3 = make_encoder_block(in_channels=2*channelscale, middle_channels=4*channelscale, 
            out_channels=4*channelscale, middle_channels2=4*channelscale, no_dsam=noupdownsam)
            self.conv4 = make_encoder_block(in_channels=4*channelscale, middle_channels=8*channelscale, 
            out_channels=8*channelscale, middle_channels2=8*channelscale, no_dsam=noupdownsam)
            self.conv5 = nn.Sequential(
                nn.Conv2d(in_channels=8*channelscale, out_channels=8*channelscale, kernel_size=3, padding=1),
                nn.InstanceNorm2d(512),
                nn.Conv2d(8*channelscale, 8*channelscale, kernel_size=(3, 3), padding=(1,1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=8*channelscale, out_channels=8*channelscale, kernel_size=3, padding=1),
                nn.InstanceNorm2d(4*channelscale),
                # nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.ReLU(inplace=True)
                )

            # self.dec5 = make_decoder_block(512, 512, 256)
            self.dec4 = make_decoder_block(16*channelscale, 8*channelscale, 4*channelscale, noupsam=noupdownsam)
            self.dec3 = make_decoder_block(8*channelscale, 4*channelscale, channelscale, noupsam=noupdownsam)
            self.dec2 = make_decoder_block(3*channelscale, 2*channelscale, channelscale//2, noupsam=noupdownsam)
            self.dec1 = nn.Sequential(
                nn.Conv2d(int(1.5*channelscale), channelscale//2, 3, padding=1), nn.ReLU(inplace=True))
            self.final = nn.Conv2d(channelscale//2, out_classes, kernel_size=1)

            #edges decoding
            self.conv15 = nn.Sequential(
                nn.Conv2d(in_channels=8*channelscale, out_channels=8*channelscale, kernel_size=3, padding=1),
                nn.InstanceNorm2d(512),
                nn.Conv2d(8*channelscale, 8*channelscale, kernel_size=(3, 3), padding=(1,1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=8*channelscale, out_channels=8*channelscale, kernel_size=3, padding=1),
                nn.InstanceNorm2d(4*channelscale),
                # nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.ReLU(inplace=True)
                )
            self.dec14 = make_decoder_block(16*channelscale, 8*channelscale, 4*channelscale, noupsam=noupdownsam)
            self.dec13 = make_decoder_block(8*channelscale, 4*channelscale, channelscale, noupsam=noupdownsam)
            self.dec12 = make_decoder_block(3*channelscale, 2*channelscale, channelscale//2, noupsam=noupdownsam)
            self.dec11 = nn.Sequential(
                nn.Conv2d(int(1.5*channelscale), channelscale//2, 3, padding=1), nn.ReLU(inplace=True))
            self.final1 = nn.Conv2d(channelscale//2, out_classes, kernel_size=1)

    def forward(self, x):
            # s(x)[:,0,:,:].unsqueeze(dim=1).expand([7,32,256,256])
            conv1 = self.conv1(x)
            conv2 = self.conv2(conv1)
            conv3 = self.conv3(conv2)
            conv4 = self.conv4(conv3)
            conv5 = self.conv5(conv4)

            # center = self.center(conv5)

            # dec5 = self.dec5(conv5)
            dec4 = self.dec4(torch.cat([conv5, conv4], 1))
            dec3 = self.dec3(torch.cat([dec4, conv3], 1))
            dec2 = self.dec2(torch.cat([dec3, conv2], 1))
            dec1 = self.dec1(torch.cat([dec2, conv1], 1))

            #edges decoding
            conv15 = self.conv15(conv4)
            dec14 = self.dec4(torch.cat([conv15, conv4], 1))
            dec13 = self.dec3(torch.cat([dec14, conv3], 1))
            dec12 = self.dec2(torch.cat([dec13, conv2], 1))
            dec11 = self.dec1(torch.cat([dec12, conv1], 1))

            return self.final(dec1), self.final1(dec11)