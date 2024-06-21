# -*- coding : UTF-8 -*-
# @file   : export_onnx.py
# @Time   : 2024-06-20 16:51
# @Author : wmz
import argparse
import logging
import os
import cv2 as cv

import numpy as np
import torch
from os.path import splitext
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from unet import UNet
from unet import SCN
from utils.SpineDataSet import SpineDataset
from utils.utils import plot_img_and_mask
import matplotlib.pyplot as plt


def export_onnx(model, input, input_names, output_names, modelname):
    model.eval()
    dynamic_axes = {
        input_names[0]: {2: 'in_width', 3: 'int_height'},
        output_names[0]: {2: 'out_width', 3: 'out_height'},
        output_names[1]: {2: 'out_width', 3: 'out_height'},
        output_names[2]: {2: 'out_width', 3: 'out_height'}
    }
    dummy_input = input
    torch.onnx.export(model, dummy_input, modelname,
                      export_params=True,
                      verbose=False,
                      opset_version=12,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=dynamic_axes)


if __name__ == '__main__':
    model = "checkpoint_epoch300.pth"
    scale = 1.0
    # in_files = ["E:/Dataset/Vesta/Landmark/pngs/verse259_RL.png"]
    # in_files = ["E:/Dataset/Vesta/Landmark/pngs/JQR_RL.png"]
    in_files = ["E:/Dataset/Spine/Landmark/pngs/sub-verse004_ct_RL.png"]
    device = "cpu"

    net = SCN(in_channels=1, num_classes=25, spatial_act="sigmoid")
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    net.load_state_dict(torch.load(model, map_location=device))
    with torch.no_grad():
        input = torch.randn(1, 1, 352, 176, device=device)
        input_names = ['input']
        output_names = ['heatmap', 'HLA', 'HSC']
        export_onnx(net, input, input_names, output_names, "vertebra_centroid.onnx")
