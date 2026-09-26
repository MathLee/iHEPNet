##################################################################################
##Our code is built on GeleNet. So in this code, GeleNet refers to our iHEPNet.##
##################################################################################
import torch
import torch.nn.functional as F

import numpy as np
import pdb, os, argparse
import time
import cv2

from model.GeleNet_models import GeleNet
from data import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
opt = parser.parse_args()

dataset_path = './dataset/test_dataset/'

model = GeleNet()
model.load_state_dict(torch.load('./models/iHEPNet_ORSSD.pth'))

model.cuda()
model.eval()

test_datasets = ['ORSSD']
#test_datasets = ['EORSSD','ORSSD','ors-4199']

for dataset in test_datasets:
    save_path = './models/iHEPNet/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/image/'
    print(dataset)
    gt_root = dataset_path + dataset + '/GT/'
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    time_sum = 0
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        time_start = time.time()
        res = model(image)
        time_end = time.time()
        time_sum = time_sum+(time_end-time_start)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res1 = np.uint8(res*255)
        cv2.imwrite(save_path+name, res1)
        if i == test_loader.size-1:
            print('Running time {:.5f}'.format(time_sum/test_loader.size))
            print('FPS {:.5f}'.format(test_loader.size / time_sum))


