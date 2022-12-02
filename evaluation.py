"""
takes path to ground truths and predicitons and computes mIOU 
testing with png images where BW pixel values are class numbers
"""

from metrics import RunningConfusionMatrix as RCM
import cv2
import scipy
import numpy as np
import argparse
import os
import imageio
VOC_COLORMAP = np.array([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                             [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                             [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                             [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                             [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                             [0, 64, 128]])

def compute_miou(gt_path, pred_path):
    CM = RCM(list(range(21)))
    a=0
    preds = os.listdir(pred_path)
    print('n preds: {}'.format(len(preds)))
    gts = os.listdir(gt_path)
    for fname in preds:
        a+=1
        print("{}/{}".format(a,len(preds)))
        # open pred and gt as numpy arrays, flatten them
        p_img = imageio.imread(os.path.join(pred_path, fname))
        p_img2=transfer(p_img)
        """
        if np.all(p_img == np.zeros(p_img.shape) ):
            print('frt')
            continue
        """
        gt_img = imageio.imread(os.path.join(gt_path, fname))
        gt_img2 = transfer(gt_img)
        CM.update_matrix(gt_img2.flatten(), p_img2.flatten())

    miou = CM.compute_current_mean_intersection_over_union()
    return miou

def transfer(name):
    mask = name
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)  # 通道转换
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    # 标签处理
    for ii, label1 in enumerate(VOC_COLORMAP):
        locations = np.all(mask == label1, axis=-1)
        label_mask[locations] = ii
    return label_mask

if __name__ == "__main__":

    VOC_HOME = 'E:/SegmentationClassAug/SegmentationClassAugrgb/'



    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_path", default=VOC_HOME, type=str)
    parser.add_argument("--pred_path", default='D:/QQ文件/out_rwrgb(1)/out_rwrgb', type=str)
    args = parser.parse_args()




    miou = compute_miou(args.gt_path, args.pred_path)
    print('miou:{}'.format(miou))