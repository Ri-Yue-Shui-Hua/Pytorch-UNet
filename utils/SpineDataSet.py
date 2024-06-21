# -*- coding : UTF-8 -*-
# @file   : SpineDataSet.py
# @Time   : 2023-10-17 16:45
# @Author : wmz
import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SpineDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = '', num_class: int = 25):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.num_class = num_class

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH))
        img_ndarray = np.asarray(pil_img)
        img_ndarray = align_16(img_ndarray)

        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))  # ensure channel，height, width order

        if not is_mask:
            img_ndarray = img_ndarray / 255

        return img_ndarray

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    @classmethod
    def load_txt(cls, filename):
        ext = splitext(filename)[1]
        coord_list = []
        if ext in ['.txt']:
            with open(filename, "r", encoding='utf-8') as f:  # 打开文本
                for data in f.readlines():
                    substr = data.replace("\n", "").split(" ")
                    c, x, y = substr
                    coord = [int(c), float(x), float(y)]
                    coord_list.append(coord)
        return coord_list

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        coord_list = self.load_txt(mask_file[0])
        img = self.load(img_file[0])
        image_shape = img.size[1], img.size[0]
        landmarks = generate_landmark(coord_list, image_shape, self.num_class)
        img = self.preprocess(img, self.scale, is_mask=False)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'heatmap': torch.as_tensor(landmarks.copy()).float().contiguous()
        }


def align_16(img_arr):
    Y, X = img_arr.shape
    new_shape = np.array([16 * np.ceil(Y / 16), 16 * np.ceil(X / 16)])
    newy, newx = new_shape.astype(int)
    align_img_arr = np.zeros((newy, newx))
    align_img_arr[:Y, :X] = img_arr
    return align_img_arr


def CenterLabelHeatMap(img_width, img_height, c_x, c_y, sigma):
    X1 = np.linspace(1, img_width, img_width)
    Y1 = np.linspace(1, img_height, img_height)
    [X, Y] = np.meshgrid(X1, Y1)
    X = X - c_x
    Y = Y - c_y
    D2 = X * X + Y * Y
    E2 = 2.0 * sigma * sigma
    Exponent = D2 / E2
    heatmap = np.exp(-Exponent)
    return heatmap


def generate_landmark(coord_list: list, image_shape: [tuple, list], num: int) -> np.ndarray:
    '''Generate one-hot landmark volume'''
    Y, X = image_shape
    new_shape = np.array([16 * np.ceil(Y / 16), 16 * np.ceil(X / 16)])
    newy, newx = new_shape.astype(int)
    landmark = np.zeros((num, *(newy, newx)), np.float32)
    classes = [coord_list[i][0] for i in range(len(coord_list))]
    index = 0
    for c in range(1, num+1):
        if c in classes:
            cx, cy = coord_list[index][1] * image_shape[1], coord_list[index][2] * image_shape[0]
            landmark[c-1, :image_shape[0], :image_shape[1]] = CenterLabelHeatMap(image_shape[1], image_shape[0], cx, cy, 21)
            index += 1
    return landmark


def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            # if mask[i, :, :].max() < 0.1:
            #     continue
            ax[i + 1].set_title(f'cls{i + 1}')
            # ax[i + 1].imshow(mask[:, :, i])
            ax[i + 1].imshow(mask[i, :, :])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    image_folder = r"E:\Dataset\Vesta\Landmark\pngs"
    label_folder = r"E:\Dataset\Vesta\Landmark\labels"
    spine_dataset = SpineDataset(images_dir=image_folder, masks_dir=label_folder)
    num = len(spine_dataset)
    train_loader = DataLoader(spine_dataset, 1)
    batch = next(iter(train_loader))
    plot_img_and_mask(batch['image'][0][0], batch['heatmap'][0])
    print(batch['image'].shape)
    print(batch['heatmap'].shape)


