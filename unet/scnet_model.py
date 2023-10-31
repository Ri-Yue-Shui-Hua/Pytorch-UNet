# -*- coding : UTF-8 -*-
# @file   : scnet_model.py
# @Time   : 2023-10-13 14:13
# @Author : wmz
import torch
import torch.nn as nn
from typing import Union, Sequence, Dict, Type
device = torch.device('cpu')
ACT = {'relu': nn.ReLU, 'leaky': nn.LeakyReLU,
       'prelu': nn.PReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid}


class LocalAppearance(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        filters: int = 64,
        dropout: float = 0.,
        mode: str = 'add',
    ):
        super().__init__()
        self.mode = mode
        self.pool = nn.AvgPool2d(2, 2, ceil_mode=True)
        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.in_conv = self.Block(in_channels, filters)
        self.enc1 = self.Block(filters, filters, dropout)
        self.enc2 = self.Block(filters, filters, dropout)
        self.enc3 = self.Block(filters, filters, dropout)
        self.enc4 = self.Block(filters, filters, dropout)
        if mode == 'add':
            self.dec3 = self.Block(filters, filters, dropout)
            self.dec2 = self.Block(filters, filters, dropout)
            self.dec1 = self.Block(filters, filters, dropout)
        else:
            self.dec3 = self.Block(2*filters, filters, dropout)
            self.dec2 = self.Block(2*filters, filters, dropout)
            self.dec1 = self.Block(2*filters, filters, dropout)
        self.out_conv = nn.Conv2d(filters, num_classes, 1, bias=False)
        nn.init.trunc_normal_(self.out_conv.weight, 0, 1e-4)

    def Block(self, in_channels, out_channels, dropout=0):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.Dropout2d(dropout, True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.Dropout2d(dropout, True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.in_conv(x)
        e1 = self.enc1(x0)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        if self.mode == 'add':
            d3 = self.dec3(self.up(e4)+e3)
            d2 = self.dec2(self.up(d3)+e2)
            d1 = self.dec1(self.up(d2)+e1)
        else:
            d3 = self.dec3(torch.cat([self.up(e4), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))

        out = self.out_conv(d1)
        return d1, out


class SCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        filters: int = 64,
        factor: int = 2,
        dropout: float = 0.,
        mode: str = 'add',
        local_act: str = None,
        spatial_act: str = 'tanh',
    ):
        super().__init__()
        self.n_classes = num_classes
        self.HLA = LocalAppearance(
            in_channels, num_classes, filters, dropout, mode)
        self.down = nn.AvgPool2d(factor, factor, ceil_mode=True)
        self.up = nn.Upsample(scale_factor=factor,
                              mode='bilinear', align_corners=True)
        self.local_act = ACT[local_act]() if local_act else None
        self.HSC = nn.Sequential(
            nn.Conv2d(filters, filters, 7, 1, 3, bias=False),
            nn.Conv2d(filters, filters, 7, 1, 3, bias=False),
            nn.Conv2d(filters, filters, 7, 1, 3, bias=False),
            nn.Conv2d(filters, num_classes, 7, 1, 3, bias=False),
        )
        self.spatial_act = ACT[spatial_act]()
        nn.init.trunc_normal_(self.HSC[-1].weight, 0, 1e-4)

    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        d1, HLA = self.HLA(x)
        if self.local_act:
            HLA = self.local_act(HLA)
        HSC = self.up(self.spatial_act(self.HSC(self.down(d1))))
        heatmap = HLA * HSC
        return heatmap, HLA, HSC


def export_onnx(model, modelname):
    with torch.no_grad():
        model.eval()
        dummy_input = torch.randn(1, 3, 1024, 1024, device=device)
        input_names = ['input']
        output_names = ['heatmap', 'HLA', 'HSC']
        torch.onnx.export(model, dummy_input, modelname,        
                      export_params=True,
                      verbose=False,
                      opset_version=12,
                      input_names=input_names,
                      output_names=output_names)
    print("export onnx model success!")


if __name__ == '__main__':
    inputs = torch.rand((1, 3, 1024, 1024))
    # net = UNet(n_channels=3, n_classes=2)
    net = SCN(in_channels=3, num_classes=25, local_act='leaky')
    export_onnx(net, "scn.onnx")
    # output = net(inputs)
    # print(output.shape)

