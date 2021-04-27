import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.utils.data as data
import functools
from torchvision.models import vgg19


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm2d(out_ch))

    def forward(self, x):

        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm2d(out_ch)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)

class AtrousConv(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.atrous_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, dilation=2, padding=2),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(),
    
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, dilation=4, padding=4),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(),
            
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, dilation=8, padding=8),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(),
            
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, dilation=16, padding=16),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(),
        )
        
    def forward(self, x):
        return self.atrous_conv(x)

class UNetSemantic2(nn.Module):
    def __init__(self, in_channels=4, latent_channels=64, out_channels=3):
        super(UNetSemantic2, self).__init__()

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Conv1 = conv_block(in_channels, filters[0])
        self.SE1 = SE_Block(filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.SE2 = SE_Block(filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.SE3 = SE_Block(filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Atrous = AtrousConv(filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_channels, kernel_size=1, stride=1, padding=0)
        self.active = torch.nn.Sigmoid()
        # self.active = torch.nn.Tanh()

    def forward(self, img, mask):

        # first_masked_img = img * (1 - mask) + mask
        # first_in = torch.cat((first_masked_img, mask), 1) 
        first_in = torch.cat((img, mask), 1) 

        e1 = self.Conv1(first_in)
        e1 = self.SE1(e1)
        e2 = self.Conv2(e1)
        e2 = self.SE2(e2)
        e3 = self.Conv3(e2)
        e3 = self.SE3(e3)
        e4 = self.Conv4(e3)
        e5 = self.Conv5(e4)

        e5 = self.Atrous(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        out = self.active(out)

        return out


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

class PerceptualNet(nn.Module):
    # https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
    def __init__(self, name = "vgg19", resize=True):
        super(PerceptualNet, self).__init__()
        blocks = []
        if name == "vgg19":
            blocks.append(vgg19(pretrained=True).features[:4].eval())
            blocks.append(vgg19(pretrained=True).features[4:9].eval())
            blocks.append(vgg19(pretrained=True).features[9:16].eval())
            blocks.append(vgg19(pretrained=True).features[16:23].eval())
        else:
            assert "wrong model name"
        
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, inputs, targets):
        if inputs.shape[1] != 3:
            inputs = inputs.repeat(1, 3, 1, 1)
            targets = targets.repeat(1, 3, 1, 1)
        inputs = (inputs-self.mean) / self.std
        targets = (targets-self.mean) / self.std
        if self.resize:
            inputs = self.transform(inputs, mode='bilinear', size=(512, 512), align_corners=False)
            targets = self.transform(targets, mode='bilinear', size=(512, 512), align_corners=False)
        loss = 0.0
        x = inputs
        y = targets
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss



