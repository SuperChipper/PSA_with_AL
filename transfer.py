import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import cv2
def decode_segmap(label_mask, plot=False):
    """
    功能：
        解码图像
    参数:
        label_mask (np.ndarray): (M,N)维度的含类别信息的矩阵.
        plot (bool, optional): 是否绘制图例.

    结果:
        (np.ndarray, optional): 解码后的色彩图.
    """
    label_colours = get_pascal_labels()
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, 21):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb


def get_pascal_labels():
    """
    Pascal VOC各类别对应的色彩标签

    结果:
        (21, 3)矩阵，含各类别对应的三通道数值信息
    """
    return np.asarray(
        [
            [0, 0, 0],
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [192, 0, 0],
            [64, 128, 0],
            [192, 128, 0],
            [64, 0, 128],
            [192, 0, 128],
            [64, 128, 128],
            [192, 128, 128],
            [0, 64, 0],
            [128, 64, 0],
            [0, 192, 0],
            [128, 192, 0],
            [0, 64, 128],
        ]
    )


def encode_segmap(mask):
    """
    功能：
        将label转换为对应的类别信息
    参数:
        mask (np.ndarray): 原始label信息.
    返回值:
        (np.ndarray): 含色彩信息的label.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)

aug_path=".\out_rw\\"
out_path = ".\out_rwrgb\\"

if os.path.exists(out_path) is not True:
    os.mkdir(out_path)


for img in os.listdir(aug_path):
    img_path = os.path.join(aug_path,img)
    img1 = imageio.imread(img_path)
    decoded = decode_segmap(img1)
    imageio.imwrite(os.path.join(out_path,img), decoded)