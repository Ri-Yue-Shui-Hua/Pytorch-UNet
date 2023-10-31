# -*- coding : UTF-8 -*-
# @file   : evaluate_coord.py
# @Time   : 2023-10-23 15:03
# @Author : wmz
import logging
import sys
import os
from pathlib import Path
from os.path import splitext
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from unet import SCN
from utils.SpineDataSet import SpineDataset
from PIL import Image
import pandas as pd


def get_files(path, suffix):
    return [os.path.join(root, file) for root, dirs, files in os.walk(path) for file in files if file.endswith(suffix)]


def realign_img(img_arr, w, h):
    return img_arr[:, :h, :w]


def predict_img(net,
                full_img,
                device,
                scale_factor=1):
    net.eval()
    w, h = full_img.size
    img = torch.from_numpy(SpineDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        heatmap, HLA, HSC = net(img)
        channel_pred = realign_img(heatmap[0], w, h)
        pred_img = channel_pred.cpu().numpy()
    return pred_img


def distance(y, x, predy,predx):
    return np.sqrt((y-predy)**2 + (x-predx)**2)


def get_recall_precision(landmark, gtlandmark):
   prenum = len(landmark)
   gtnum = len(gtlandmark)
   # 当指定类别小于一定距离时才认为是预测正确
   ijkLandmarks_arr = np.array(landmark)
   pred_class = list(ijkLandmarks_arr[:,0])
   gtijkLandmarks_arr = np.array(gtlandmark)
   gt_class = list(gtijkLandmarks_arr[:,0])
   merge_list = pred_class + gt_class
   print(merge_list)
   merge_set = set(merge_list)
   print(merge_set)
   intersect = set(pred_class).intersection(gt_class)
   print(intersect)
   intersect_list = list(intersect)
   print(intersect_list)
   # 重合的都是可能的召回 TP/(TP+FN)
   tp = 0
   fn = 0
   for value in intersect_list:
      preindex = np.where(ijkLandmarks_arr[:, 0] == value)
      gtindex = np.where(gtijkLandmarks_arr[:, 0] == value)
      y, x = gtijkLandmarks_arr[gtindex[0][0],  1:]
      predy,predx = ijkLandmarks_arr[preindex[0][0], 1:]
      dist = distance(y, x, predy, predx)
      print(dist)
      if dist > 22:
         continue
      tp += 1
   recall = tp/gtnum
   precision = tp/prenum
   return tp, gtnum, recall, prenum, precision, pred_class, gt_class


def load_label_txt(filename, w, h):
    ext = splitext(filename)[1]
    coord_list = []
    if ext in ['.txt']:
        with open(filename, "r", encoding='utf-8') as f:  # 打开文本
            for data in f.readlines():
                substr = data.replace("\n", "").split(" ")
                c, x, y = substr
                coord = [int(c), float(x)*w, float(y)*h]
                coord_list.append(coord)
        return coord_list


def save_to_csv(bbox_info, csv_name):
    pd.DataFrame.from_dict(bbox_info).to_csv(csv_name, index=False)


if __name__ == "__main__":
    dir_img = Path("E:/Dataset/Spine/Landmark/pngs")
    dir_mask = Path("E:/Dataset/Spine/Landmark/labels")
    # dir_img = Path('/home/jmed/wmz/DataSet/Spine2D/Landmark/pngs/')
    # dir_mask = Path('/home/jmed/wmz/DataSet/Spine2D/Landmark/labels/')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net = SCN(in_channels=1, num_classes=25, spatial_act="sigmoid")
    net.to(device=device)
    checkpoint_model = "checkpoint_epoch422.pth"
    net.load_state_dict(torch.load(checkpoint_model, map_location=device))
    logging.info(f'Model loaded from {checkpoint_model}')
    png_suffix = "_RL.png"
    txt_suffix = "_RL.txt"
    file_list = get_files(dir_img, png_suffix)
    thresh = 0.4
    all_tp = 0
    all_gtnum = 0
    all_prenum = 0
    fid_list = []
    metric_list = []
    pred_class_list = []
    gt_class_list = []
    metric_info = {}
    for i, filename in enumerate(file_list):
        logging.info(f'\nPredicting image {filename} ...')
        file_id = os.path.basename(filename).replace(png_suffix, "")
        img = Image.open(filename)

        masks = predict_img(net=net,
                           full_img=img,
                            device=device,
                           scale_factor=1.0)
        channels = masks.shape[0]
        ijkLandmarks = []
        for i in range(channels):
            max_value = np.max(masks[i, :, :])
            if max_value > thresh:
                y, x = np.where(masks[i, :, :] == max_value)
                print(x, y)
                ijkLandmarks.append([i + 1, x.mean(), y.mean()])
        w, h = img.size
        label_file = filename.replace("pngs", "labels").replace(png_suffix, txt_suffix)
        if len(ijkLandmarks) == 0:
            print(file_id, " no pred points")
            continue
        file_id = os.path.basename(filename).replace(png_suffix, "")
        fid_list.append(file_id)
        gtijkLandmarks = load_label_txt(label_file, w, h)
        tp, gtnum, recall, prenum, precision, pred_class, gt_class = get_recall_precision(ijkLandmarks, gtijkLandmarks)
        metric_list.append([tp, gtnum, recall, prenum, precision])
        pred_class_list.append(str(pred_class))
        gt_class_list.append(str(gt_class))
        print("tp, gtnum, recall, prenum, precision \n", tp, gtnum, recall, prenum, precision)
        all_tp += tp
        all_gtnum += gtnum
        all_prenum += prenum
        # 计算定位平均距离误差
        # exit(0)
    all_recall = all_tp / all_gtnum
    all_precision = all_tp / all_prenum
    print("all_tp,all_gtnum,all_prenum \n", all_tp, all_gtnum, all_prenum)
    print("all_recall, all_precision \n", all_recall, all_precision)
    csv_header = ['fid', 'tp', 'gtnum', 'recall', 'prenum', 'precision', 'pred_class', "gt_class"]
    metric_info.setdefault(csv_header[0], fid_list)
    metric_info_arr = np.array(metric_list)
    for idx in range(1, len(csv_header) - 2):
        metric_info.setdefault(csv_header[idx], metric_info_arr[:, idx - 1])
    metric_info.setdefault(csv_header[-2], pred_class_list)
    metric_info.setdefault(csv_header[-1], gt_class_list)
    csv_file = f"test_model_{thresh}.csv"
    save_to_csv(metric_info, csv_file)










