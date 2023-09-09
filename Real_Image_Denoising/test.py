import cv2
import os
import argparse
from tqdm import tqdm
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from models import *
from utils import *
import torchvision.utils as utils
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from PIL import ImageFilter
from imageio import imwrite
import utilspackage

from utilspackage.loader import get_validation_data
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from skimage import img_as_float32, img_as_ubyte

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="WDEN")
parser.add_argument("--num_of_layers", type=int, default=16, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument('--input_dir', default='data/sidd/val/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='fig_result/denoising/sidd/', type=str, help='Directory for results')
parser.add_argument('--save_images', type=bool, default=False, help='Save denoised images in result directory')
opt = parser.parse_args()

utilspackage.mkdir(opt.result_dir)

test_dataset = get_validation_data(opt.input_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)


def normalize(data):
    return data/255.

def main():
    toTensor = transforms.Compose([transforms.ToTensor()])
    toPILImg = transforms.ToPILImage()

    # Build model
    print('Loading model ...\n')
    net = WDENet(channels=3)
    device_ids = [0]

    #model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model = net.cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'WDENet_real_pre.pth'))) # Input model's path files. 30.6419
    model.eval()
    # load data info
    print('Loading data info ...\n')
    # process data
    file_handle = open('result.txt', mode='w')
    with torch.no_grad():
        psnr_val_rgb = []
        ssim_val_rgb = []
        for ii, data_test in enumerate(tqdm(test_loader), 0):
            rgb_gt = data_test[0].numpy().squeeze().transpose((1, 2, 0))
            rgb_noisy = data_test[1].cuda()
            filenames = data_test[2]

            rgb_restored = model(rgb_noisy)
            rgb_restored = torch.clamp(rgb_restored, 0, 1).cpu().numpy().squeeze().transpose((1, 2, 0))
            psnr_data = psnr_loss(rgb_restored, rgb_gt)
            ssim_data = ssim_loss(rgb_restored, rgb_gt, multichannel=True)
            psnr_val_rgb.append(psnr_data)
            ssim_val_rgb.append(ssim_data)
            file_handle.write("filename: %s, psnr: %f, ssim: %f\n" % (filenames[0], psnr_data, ssim_data))

            if True:
                utilspackage.save_img(os.path.join(opt.result_dir, filenames[0]), img_as_ubyte(rgb_restored))
    file_handle.close()
    psnr_val_rgb = sum(psnr_val_rgb) / len(test_dataset)
    ssim_val_rgb = sum(ssim_val_rgb) / len(test_dataset)
    print("PSNR: %f, SSIM: %f " % (psnr_val_rgb, ssim_val_rgb))


if __name__ == "__main__":
    main()
