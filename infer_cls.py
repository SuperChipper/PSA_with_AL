
import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
import voc12.data
import scipy.misc
import imageio
import importlib
from torch.utils.data import DataLoader
import torchvision
from tool import imutils, pyutils
import argparse
from PIL import Image
import torch.nn.functional as F
import os.path
import cv2
#import skimage
#from skimage.segmentation import slic, mark_boundaries
#from skimage import io
from networkx.linalg import adj_matrix
from skimage.future import graph
from skimage import segmentation
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from multiprocessing import Pool
from tqdm import tqdm
import os
import imageio
if __name__ == '__main__':


    ###读入标签
    # 标签中每个RGB颜色的值
    VOC_COLORMAP = np.array([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                             [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                             [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                             [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                             [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                             [0, 64, 128]])
    # 标签其标注的类别
    VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                   'diningtable', 'dog', 'horse', 'motorbike', 'person',
                   'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

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
        return label_mask


    # 彩色图路径

    out_path = ".\\VOCdevkit\\VOC2012\\SegmentationClassAugrgb\\"
    if os.path.exists(out_path) is not True:
        os.mkdir(out_path)

    # 处理txt中的对应图像
    txt_path = '.\\voc12\\train_aug.txt'
    # 标签所在的文件夹
    label_file_path = '.\\VOCdevkit\\VOC2012\\SegmentationClassAug'
    # 处理后的标签保存的地址
    gray_save_path = '.\\'

    # with open(txt_path, 'r') as f:
    #     file_names = f.readlines()
    #     for name in tqdm(file_names):
    #         name = name.strip('\n')  # 去掉换行符
    #         name = name.split(' ')[0][-15:-4]
    #         label_name = name + '.png'  # label文件名
    #         label_url = os.path.join(label_file_path, label_name)
    #
    #         mask = cv2.imread(label_url)
    #         mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)  # 通道转换
    #         mask = mask.astype(int)
    #         label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    #         # 标签处理
    #         for ii, label in enumerate(VOC_COLORMAP):
    #             locations = np.all(mask == label, axis=-1)
    #             label_mask[locations] = ii
    #         # 标签保存
    #         cv2.imwrite(gray_save_path + label_name, label_mask)
    #
    #   # 初始化slic项，超像素平均尺寸20（默认为10），平滑因子20


    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="res38_cls.pth",required=False, type=str)
    parser.add_argument("--network", default="network.resnet38_cls", type=str)
    parser.add_argument("--infer_list", default="voc12/train_aug1.txt", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--voc12_root",default="VOCdevkit\VOC2012", required=False, type=str)
    parser.add_argument("--low_alpha", default=4, type=int)
    parser.add_argument("--high_alpha", default=32, type=int)
    parser.add_argument("--out_cam", default="cam", type=str)##None->Yes
    parser.add_argument("--out_la_crf", default="out_la_crf", type=str)
    parser.add_argument("--out_ha_crf", default="out_ha_crf", type=str)
    parser.add_argument("--out_cam_pred", default="cam_pred", type=str)

    args = parser.parse_args()

    model = getattr(importlib.import_module(args.network), 'Net')()
    model.load_state_dict(torch.load(args.weights))

    model.eval()
    model.cuda()

    infer_dataset = voc12.data.VOC12ClsDatasetMSF(args.infer_list, voc12_root=args.voc12_root,
                                                   scales=(1, 0.5, 1.5, 2.0),
                                                   inter_transform=torchvision.transforms.Compose(
                                                       [np.asarray,
                                                        model.normalize,
                                                        imutils.HWC_to_CHW]))

    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    n_gpus = torch.cuda.device_count()
    model_replicas = torch.nn.parallel.replicate(model, list(range(n_gpus)))
    ###超像素分割
    length=len(infer_data_loader)
    procbar=tqdm(total=length)
    #p=Pool(processes=None)
    for iter, (img_name, img_list, label) in enumerate(infer_data_loader):
        img_name = img_name[0]; label = label[0]

        img_path = voc12.data.get_img_path(img_name, args.voc12_root)
        orig_img = np.asarray(Image.open(img_path))
        orig_img_size = orig_img.shape[:2]
        ###加入自己的判别式
        ###设置规则网格
        slic = cv2.ximgproc.createSuperpixelSLIC(orig_img, region_size=30, ruler=20.0)
        slic.iterate(10)  # 迭代次数，越大效果越好
        mask_slic = slic.getLabelContourMask()  # 获取Mask，超像素边缘Mask==1
        label_slic = slic.getLabels()  # 获取超像素标签
        number_slic = slic.getNumberOfSuperpixels()  # 获取超像素数目
        mask_inv_slic = cv2.bitwise_not(mask_slic)
        np.argmin(label_slic)
        # label_slic=segmentation.slic(orig_img,compactness=30,n_segments=400)
        g = graph.RAG(label_slic)
        slic_order = list(g.nodes)  ###生成的superpixel序号不规则，用此列表记录
        adj = adj_matrix(g).todense()

        w = np.shape(label_slic)[0]
        h = np.shape(label_slic)[1]
        slic = [[] for i in range(number_slic)]
        for i in range(w):
            for j in range(h):
                value = label_slic[i][j]
                slic[value].append((i, j))

        img_slic = cv2.bitwise_and(orig_img, orig_img, mask=mask_inv_slic)  # 在原图上绘制超像素边界
        #mt.PIS(img_slic)  # 显示图像


        ###生成CAM
        def _work(i, img):
            with torch.no_grad():
                with torch.cuda.device(i % n_gpus):
                    cam = model_replicas[i % n_gpus].forward_cam(img.cuda())
                    cam = F.upsample(cam, orig_img_size, mode='bilinear', align_corners=False)[0]
                    cam = cam.cpu().numpy() * label.clone().view(20, 1, 1).numpy()
                    if i % 2 == 1:
                        cam = np.flip(cam, axis=-1)
                    return cam


        thread_pool = pyutils.BatchThreader(_work, list(enumerate(img_list)),
                                            batch_size=2, prefetch_size=0, processes=args.num_workers)
        ###batch_size=12->2
        cam_list = thread_pool.pop_results()
        sum_cam = np.sum(cam_list, axis=0)
        norm_cam = sum_cam / (np.max(sum_cam, (1, 2), keepdims=True) + 1e-5)
        cam_dict = {}

        ###CAM最大概率类别的热力图
        # if args.out_cam_pred is not None:
        #     bg_score = [np.ones_like(norm_cam[0]) * 0.2]
        #     aaa = np.concatenate((bg_score, norm_cam))
        #     pred = np.argmax(np.concatenate((bg_score, norm_cam)), 0)
        #     cam_ssum = np.zeros(20)
        #     j = 0
        #     while (j < 20):
        #         cam_ssum[j] = np.sum(norm_cam[j])
        #         j += 1
        #
        #     cam1 = norm_cam[np.argmax(cam_ssum)]
        #
        #     cam_img = np.uint8(cam1 * 255)
        #     heatmap = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
        #     result = heatmap * 0.3 + orig_img * 0.5
        #     imageio.imsave(os.path.join(args.out_cam_pred, img_name + '.png'), result)  # pred.astype(np.uint8))
        #     ###scipy.misc.imsave->imageio.imsave
        ###计算分数
            #超像素内部均值
        mean_slic = np.zeros([number_slic, 20])
        norm_cam1 = norm_cam.transpose(1, 2, 0)
        norm_cam_copy=np.zeros_like(norm_cam1)
        for i in range(number_slic):
            length_slic_i = len(slic[i])

            sum_slic_i = np.zeros(20)
            coordinates=slic[i]

            #a=np.sum(norm_cam1[x,y],axis=0)
            try:
                x, y = np.array(coordinates).T
                sum_slic_i=np.sum(norm_cam1[x,y],axis=0)
            #sum_slic_i = np.zeros(20)
            #for j in range(length_slic_i):
                #coordinate = slic[i][j]
                #x,y= coordinate
                #a = norm_cam1[x][y]
                #sum_slic_i += a
                mean_slic[i] = sum_slic_i / length_slic_i
            except:
                mean_slic[i]=0

            #超像素内部方差
        std_slic = np.zeros(number_slic)
        mean_slic1 = np.zeros(number_slic)
        score = np.zeros(number_slic)
        argmax_mean_slic = np.argmax(mean_slic, axis=1)
        for i in range(number_slic):
            length_slic_i = len(slic[i])
            coordinates = slic[i]
            try:
                x, y = np.array(coordinates).T
                a = norm_cam1[x,y]
                s = np.sum(np.multiply(a - mean_slic[i], a - mean_slic[i]),axis=0)
            #for j in range(length_slic_i):
                #coordinate = slic[i][j]
                #x,y = coordinate
                #a = norm_cam1[x][y]
                #s=np.sum(np.dot(a-mean_slic[i],a-mean_slic[i]))
                #std = (s ** 0.5) / length_slic_i

                std_slic[i] = np.sqrt(np.sum(s*s)/length_slic_i)
                mean_slic1[i] = max(mean_slic[i])
            except:
                std_slic[i]=0
                mean_slic1[i]=0
            score[i] = (-10 * std_slic[i] - abs(mean_slic1[i] - 0.3))

        # # 显示出confidence area
        # test_vector=np.zeros(20)
        # for i in range(w):
        #     for j in range(h):
        #
        #      if (np.argmax(cam_ssum) == np.argmax(norm_cam1[i][j])):
        #          if(1-np.array_equal(norm_cam1[i][j],test_vector)):
        #              result[i][j][0] = 0
        #              result[i][j][1] = 255
        #              result[i][j][2] = 0

        ###判断超像素需要标注的程度,在原图上标注其位置
            ###不考虑nan超像素
        for i in range(np.shape(score)[0]):
            if (np.isnan(score[i])):
                score[i] = score[np.argsort(score)[2]]
        slic_selected1 = slic[np.argmax(score)]

        # for i in range(len(slic_selected1)):
        #     coordinate = slic_selected1[i]
        #     x = coordinate[0]
        #     y = coordinate[1]
        #     result[x][y][0] = 255
        #     result[x][y][1] = 0
        #     result[x][y][2] = 0

            ###寻找selected_slic周围能够合并的超像素

        index = slic_order.index(np.argmax(score))
        around_slic = np.nonzero(adj[index])[1]
        around_slic2 = np.zeros_like(around_slic)
        for i in range(np.shape(around_slic)[0]):
            around_slic2[i] = slic_order[around_slic[i]]

        # for i in range(np.shape(around_slic2)[0]):
        #     if (abs(score[around_slic2[i]] - score[np.argmax(score)]) <= 0.1):
        #         for j in range(len(slic[around_slic2[i]])):
        #             coordinate = slic[around_slic2[i]][j]
        #             x = coordinate[0]
        #             y = coordinate[1]
        #             result[x][y][0] = 0
        #             result[x][y][1] = 255
        #             result[x][y][2] = 0
            #可视化
        # imageio.imsave(os.path.join(args.out_cam_pred, img_name + 'mark' + '.png'),
        #                result)  # pred.astype(np.uint8))
        ###模拟标注
        label_name = img_name + '.png'  # label文件名
        #label_url = os.path.join(label_file_path, label_name)

        #img1 = imageio.imread(label_url)
        #decoded = decode_segmap(img1)
        #imageio.imsave(os.path.join(out_path,label_name), decoded)
        mask = cv2.imread(os.path.join(out_path, label_name))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)  # 通道转换
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
            # 标签处理
        for ii, label1 in enumerate(VOC_COLORMAP):
            locations = np.all(mask == label1, axis=-1)
            label_mask[locations] = ii

            # 在标签图上找到对应的区域，进行标注
        atl_save_path="label_dict"
        if not os.path.isdir(atl_save_path):
            os.mkdir(atl_save_path)
        
        for i in range(len(slic_selected1)):
            coordinate = slic_selected1[i]
            x = coordinate[0]
            y = coordinate[1]
            #print(label_mask[x][y])
            if (label_mask[x][y] != 0):
                ##该点处的cam数值为[0,0,....0]
                #norm_cam1[x][y] = np.zeros(20)
                norm_cam_copy[x][y][label_mask[x][y] - 1] = 1

            #else:
                ##该点处的cam数值为[0,0,..1..0]
                #norm_cam1[x][y] = np.zeros(20)
                #norm_cam1[x][y][label_mask[x][y] - 1] = 1

        for i in range(np.shape(around_slic2)[0]):
            if (abs(score[around_slic2[i]] - score[np.argmax(score)]) <= 0.1):
                for j in range(len(slic[around_slic2[i]])):

                    x,y= slic[around_slic2[i]][j]

                    #norm_cam1[x][y] = np.zeros(20)
                    if (label_mask[x][y] != 0):
                        norm_cam_copy[x][y][label_mask[x][y] - 1] = 1
            # 可视化
        copy_dict={}
        norm_cam_copy = norm_cam_copy.transpose(2, 0, 1)
        norm_cam1 = norm_cam1.transpose(2, 0, 1)
        # imageio.imsave(os.path.join(args.out_cam_pred, img_name + 'mark11' + '.png'),
        #                255 * norm_cam1[np.argmax(cam_ssum)])

        ##输出cam
        for i in range(20):
            #print(label[i])
            if label[i] > 1e-5:
                cam_dict[i] = norm_cam1[i]
                copy_dict[i] = norm_cam_copy[i]
        np.save(os.path.join(atl_save_path, img_name + '.npy'), copy_dict)
        if args.out_cam is not None:
            np.save(os.path.join(args.out_cam, img_name + '.npy'), cam_dict)

        def _crf_with_alpha(cam_dict, alpha):
            v = np.array(list(cam_dict.values()))
            bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
            bgcam_score = np.concatenate((bg_score, v), axis=0)
            crf_score = imutils.crf_inference(orig_img, bgcam_score, labels=bgcam_score.shape[0])

            n_crf_al = dict()

            n_crf_al[0] = crf_score[0]
            for i, key in enumerate(cam_dict.keys()):
                n_crf_al[key + 1] = crf_score[i + 1]

            return n_crf_al


        if args.out_la_crf is not None:
            crf_la = _crf_with_alpha(cam_dict, args.low_alpha)
            np.save(os.path.join(args.out_la_crf, img_name + '.npy'), crf_la)

        if args.out_ha_crf is not None:
            crf_ha = _crf_with_alpha(cam_dict, args.high_alpha)
            np.save(os.path.join(args.out_ha_crf, img_name + '.npy'), crf_ha)
        procbar.update(1)






