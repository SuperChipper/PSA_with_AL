import torch
import torchvision
from tool import imutils

import argparse
import importlib
import numpy as np
import cv2
import voc12.data
from torch.utils.data import DataLoader
import imageio
import torch.nn.functional as F
import os.path
from tqdm import tqdm


def get_indices_in_radius(height, width, radius):
    search_dist = []
    for x in range(1, radius):
        search_dist.append((0, x))

    for y in range(1, radius):
        for x in range(-radius + 1, radius):
            if x * x + y * y < radius * radius:
                search_dist.append((y, x))

    full_indices = np.reshape(np.arange(0, height * width, dtype=np.int64),
                              (height, width))
    radius_floor = radius - 1
    cropped_height = height - radius_floor
    cropped_width = width - 2 * radius_floor

    indices_from = np.reshape(full_indices[:-radius_floor, radius_floor:-radius_floor], [-1])

    indices_from_to_list = []

    for dy, dx in search_dist:
        indices_to = full_indices[dy:dy + cropped_height, radius_floor + dx:radius_floor + dx + cropped_width]
        indices_to = np.reshape(indices_to, [-1])

        indices_from_to = np.stack((indices_from, indices_to), axis=1)

        indices_from_to_list.append(indices_from_to)

    concat_indices_from_to = np.concatenate(indices_from_to_list, axis=0)

    return concat_indices_from_to


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="res_aff_psa_epoch7.pth", type=str)
    parser.add_argument("--network", default="network.resnet38_aff", type=str)
    parser.add_argument("--infer_list", default="voc12/train_aug1.txt", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--cam_dir", default="cam", type=str)
    parser.add_argument("--voc12_root", default=".\VOCdevkit\VOC2012", type=str)
    parser.add_argument("--alpha", default=16, type=int)
    parser.add_argument("--out_rw", default="out_rw", type=str)
    parser.add_argument("--out_rw_mark", default="out_rw_mark", type=str)
    parser.add_argument("--beta", default=8, type=int)
    parser.add_argument("--logt", default=8, type=int)

    args = parser.parse_args()

    model = getattr(importlib.import_module(args.network), 'Net')()

    model.load_state_dict(torch.load(args.weights))

    model.eval()
    model.cuda()

    infer_dataset = voc12.data.VOC12ImageDataset(args.infer_list, voc12_root=args.voc12_root,
                                                 transform=torchvision.transforms.Compose(
                                                     [np.asarray,
                                                      model.normalize,
                                                      imutils.HWC_to_CHW]))
    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    #alt_path = "label_dict\label_dict"
    for iter, (name, img) in tqdm(enumerate(infer_data_loader)):

        name = name[0]
        # print(iter)

        orig_shape = img.shape
        padded_size = (int(np.ceil(img.shape[2] / 8) * 8), int(np.ceil(img.shape[3] / 8) * 8))

        p2d = (0, padded_size[1] - img.shape[3], 0, padded_size[0] - img.shape[2])
        img = F.pad(img, p2d)

        dheight = int(np.ceil(img.shape[2] / 8))
        dwidth = int(np.ceil(img.shape[3] / 8))

        cam = np.load(os.path.join(args.cam_dir, name + '.npy'), allow_pickle=True).item()
        #atl = np.load(os.path.join(alt_path, name + '.npy'), allow_pickle=True).item()

        cam_full_arr = np.zeros((21, orig_shape[2], orig_shape[3]), np.float32)
        #atl_full_arr = np.zeros((20, orig_shape[2], orig_shape[3]), np.float32)
        for k, v in cam.items():
            cam_full_arr[k + 1] = v

        # for k, v in cam.item():
        # cam_full_arr[k+1] = v
        cam_full_arr[0] = (1 - np.max(cam_full_arr[1:], (0), keepdims=False)) ** args.alpha
        cam_full_arr = np.pad(cam_full_arr, ((0, 0), (0, p2d[3]), (0, p2d[1])), mode='constant')

        with torch.no_grad():
            aff_mat = torch.pow(model.forward(img.cuda(), True), args.beta)

            trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
            for _ in range(args.logt):
                trans_mat = torch.matmul(trans_mat, trans_mat)

            cam_full_arr = torch.from_numpy(cam_full_arr)
            cam_full_arr = F.avg_pool2d(cam_full_arr, 8, 8)

            cam_vec = cam_full_arr.view(21, -1)

            cam_rw = torch.matmul(cam_vec.cuda(), trans_mat)
            cam_rw = cam_rw.view(1, 21, dheight, dwidth)

            cam_rw = torch.nn.Upsample((img.shape[2], img.shape[3]), mode='bilinear')(cam_rw)
            _, cam_rw_pred = torch.max(cam_rw, 1)

            res = np.uint8(cam_rw_pred.cpu().data[0])[:orig_shape[2], :orig_shape[3]]
            imageio.imsave(os.path.join(args.out_rw, name + '.png'), res)
            #cv2.imshow('origin',res)
            position = "position"
            label_position = "label_mask"
            mark=np.load(os.path.join(position, name + '.npy'))
            label_mask=np.load(os.path.join(label_position, name + '.npy'))
            choice=np.zeros(20)
            for i in range(20):
                choice[i]=np.sum(label_mask[mark==1]==i+1)
            #cv2.imshow('mask',label_mask)
            label_mask[label_mask!=np.argmax(choice)+1]=0
            #cv2.imshow('label',label_mask)
            #cv2.waitKey(0)
            res[mark==1]=label_mask[mark==1]
            # for k, v in atl.items():
            #     res[v ==1] = k + 1
            #cv2.imshow('marked', res)
            #cv2.waitKey(0)
            imageio.imsave(os.path.join(args.out_rw_mark, name + '.png'), res)
            #print("ok")
