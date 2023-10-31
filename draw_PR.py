# -*- coding : UTF-8 -*-
# @file   : draw_PR.py
# @Time   : 2023-10-30 10:07
# @Author : wmz
import numpy as np
import matplotlib.pyplot as plt


def draw_precision_recall():
    thresh = [0.2, 0.3, 0.4, 0.5, 0.6]
    precision = [0.8753, 0.9147, 0.9376, 0.9525, 0.9636]
    recall = [0.9511, 0.9481, 0.9387, 0.9222, 0.8909]
    plt.plot(precision, recall, "ob-", label="recall")
    plt.xlabel("precision")
    plt.legend()
    plt.grid(1)
    plt.show()


if __name__ == "__main__":
    draw_precision_recall()





