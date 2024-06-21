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


def visualize_locate(image_path, point_list, thickness=4, point_color=(0, 0, 255)):
    if len(point_list) == 0:
        return
    image = cv.imread(image_path)
    point_size = 1
    # point_color BGR
    # thickness可以为 0 、4、8
    for point in point_list:
        cv.circle(image, point, point_size, point_color, thickness)

    cv.namedWindow("image", 0)
    cv.imshow('image', image)
    cv.waitKey()
    cv.destroyAllWindows()


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


def get_output_filenames(args):
    def _generate_name(fn):
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


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


def visualize_predict(masks, file_name, thresh):
    channels = masks.shape[0]
    point_list = []
    for i in range(channels):
        max_value = np.max(masks[i, :, :])
        print(i+1, max_value)
        if max_value > thresh:
            y, x = np.where(masks[i, :, :] == max_value)
            # print(x, y)
            point_list.append([int(x.mean()), int(y.mean())])
    visualize_locate(file_name, point_list)
    return point_list


def plot_channel_max(maxAP_value, maxRL_value=None):
    class_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    plt.plot(class_idx, maxAP_value, "ob-", label="RL")
    # plt.plot(class_idx, maxRL_value, "or-", label="RL")
    plt.xlabel("cls_idx")
    x_ticks = np.arange(0, 26, 1)
    plt.xticks(x_ticks)
    plt.legend()
    plt.grid(1)
    plt.show()


if __name__ == '__main__':
    model = "checkpoint_epoch90.pth"
    scale = 1.0
    # in_files = ["E:/Dataset/Vesta/Landmark/pngs/verse259_RL.png"]
    # in_files = ["E:/Dataset/Vesta/Landmark/pngs/JQR_RL.png"]
    in_files = ["E:/Dataset/Spine/Landmark/pngs/sub-verse004_ct_AP.png"]

    net = SCN(in_channels=1, num_classes=25, spatial_act="sigmoid")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    net.load_state_dict(torch.load(model, map_location=device))
    thresh = 0.2

    for i, filename in enumerate(in_files):
        logging.info(f'\nPredicting image {filename} ...')
        img = Image.open(filename)

        masks = predict_img(net=net,
                           full_img=img,
                            device=device,
                           scale_factor=scale)
        channels = masks.shape[0]
        max_value_list = []
        for i in range(channels):
            max_value = np.max(masks[i, :, :])
            max_value_list.append(max_value)
        plot_channel_max(max_value_list)

        w, h = img.size
        label_file = filename.replace("pngs", "labels").replace(".png", ".txt")
        gtijLandmarks = load_label_txt(label_file, w, h)
        gtijLandmarks = [[int(landmark[1]), int(landmark[2])] for landmark in gtijLandmarks]
        visualize_locate(filename, gtijLandmarks, point_color=(0, 255, 255))
        pred_landmarks = visualize_predict(masks, filename, thresh)

        exit(0)
        # if not args.no_save:
        #     for id in range(net.n_classes):
        #         out_filename = out_files + f"ch_{id}.png"
        #         result = mask_to_image(masks[id])
        #         result.save(out_filename)
        #         logging.info(f'Mask saved to {out_filename}')

        # if args.viz:
        #     logging.info(f'Visualizing results for image {filename}, close to continue...')
        #     plot_img_and_mask(img, masks)
