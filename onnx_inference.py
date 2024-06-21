# -*- coding : UTF-8 -*-
# @file   : onnx_inference.py
# @Time   : 2024-06-20 17:42
# @Author : wmz
import os
import onnxruntime as ort
import time
from PIL import Image
import logging
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def print_onnx_info(onnx_path):
    provider = "CPUExecutionProvider"
    onnx_session = ort.InferenceSession(onnx_path, providers=[provider])

    print("----------------- 输入部分 -----------------")
    input_tensors = onnx_session.get_inputs()  # 该 API 会返回列表
    for input_tensor in input_tensors:  # 因为可能有多个输入，所以为列表

        input_info = {
            "name": input_tensor.name,
            "type": input_tensor.type,
            "shape": input_tensor.shape,
        }
        print(input_info)

    print("----------------- 输出部分 -----------------")
    output_tensors = onnx_session.get_outputs()  # 该 API 会返回列表
    for output_tensor in output_tensors:  # 因为可能有多个输出，所以为列表

        output_info = {
            "name": output_tensor.name,
            "type": output_tensor.type,
            "shape": output_tensor.shape,
        }
        print(output_info)


def align_16(img_arr):
    Y, X = img_arr.shape
    new_shape = np.array([16 * np.ceil(Y / 16), 16 * np.ceil(X / 16)])
    newy, newx = new_shape.astype(int)
    align_img_arr = np.zeros((newy, newx))
    align_img_arr[:Y, :X] = img_arr
    return align_img_arr


def realign_img(img_arr, w, h):
    return img_arr[:, :h, :w]


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


if __name__ == '__main__':
    # 创建一个InferenceSession的实例，并将模型的地址传递给该实例
    # sess = onnxruntime.InferenceSession('kneeSeg.onnx')
    # ONNX_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    print_onnx_info('vertebra_centroid.onnx')
    opt = ort.SessionOptions()
    sess = ort.InferenceSession('vertebra_centroid.onnx', opt, providers=["CPUExecutionProvider"])
    # 调用实例sess的方法进行推理
    input_name = sess.get_inputs()[0].name
    output_1_name = sess.get_outputs()[0].name
    output_2_name = sess.get_outputs()[1].name
    output_3_name = sess.get_outputs()[2].name
    output_name = [output_1_name, output_2_name, output_3_name]
    print(input_name)
    print(output_1_name)
    print(output_2_name)
    print(output_3_name)
    # in_files = ["E:/Dataset/Spine/Landmark/pngs/sub-verse004_ct_RL.png"]
    in_files = ["Body1-1_RL.png"]
    thresh = 0.2
    for i, filename in enumerate(in_files):
        logging.info(f'\nPredicting image {filename} ...')
        img = Image.open(filename) # pil_img
        w, h = img.size
        img_ndarray = np.asarray(img)
        img_ndarray = align_16(img_ndarray)
        img_ndarray = img_ndarray[np.newaxis, ...]
        img_ndarray = img_ndarray / 255
        img_arr = img_ndarray
        img_arr = img_arr[np.newaxis, ...]
        img_arr = img_arr.astype(np.float32)
        start = time.time()
        outputs = sess.run(output_name, {input_name: img_arr})
        heatmap = outputs[0][0]
        channel_pred = realign_img(heatmap, w, h)
        shape = heatmap.shape
        print(shape)
        mark_pred = channel_pred
        end = time.time()
        average = (end - start) / 1.0
        print("average time cost:", average, "s")
        channels = mark_pred.shape[0]
        max_value_list = []
        for i in range(channels):
            max_value = np.max(mark_pred[i, :, :])
            max_value_list.append(max_value)
        plot_channel_max(max_value_list)
        pred_landmarks = visualize_predict(mark_pred, filename, thresh)

