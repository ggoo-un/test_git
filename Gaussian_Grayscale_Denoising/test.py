import cv2
import os
from os import path as osp
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import *
from utils import *
import torchvision.utils as utils
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from PIL import ImageFilter
from imageio import imwrite
import time
from collections import OrderedDict

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="WDENet")
parser.add_argument("--num_of_layers", type=int, default=16, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--test_data", type=str, default='Set12', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=15, help='noise level used on test set')
opt = parser.parse_args()

def normalize(data):
    return data/255.

def main():
    toTensor = transforms.Compose([transforms.ToTensor()])
    toPILImg = transforms.ToPILImage()

    test_results = OrderedDict()
    test_results['runtime'] = []  # 순서 있는 딕셔너리 생성
    noiseL = 'noise_' + str(int(opt.test_noiseL))
    logs_n = 'WDENet_' + str(int(opt.test_noiseL)) + '.pth'

    # Build model
    print('Loading model ...\n')
    net = WDENet(channels=1)
    device_ids = [0]

    #model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model = net.cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, noiseL, logs_n))) # Input model's path files.
    model.eval()
    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('data', opt.test_data, '*.png'))
    files_source.sort()
    # process data
    psnr_test = 0
    ssim_test = 0
    noise_psnr_test, noise_ssim_test = 0, 0

    start = torch.cuda.Event(enable_timing=True)  # gpu run time 계산
    end = torch.cuda.Event(enable_timing=True)

    for f in files_source:
        # image
        Img = cv2.imread(f)
        Img = normalize(np.float32(Img[:,:,0]))
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        ISource = torch.Tensor(Img)

        # noise
        noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL / 255.)
        # noisy image
        INoisy = ISource + noise

        ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())
        with torch.no_grad():  # this can save much memory
            start.record()
            Out = model(INoisy)
            end.record()
            torch.cuda.synchronize()
            test_results['runtime'].append(start.elapsed_time(end))

            Out = torch.clamp(Out, 0., 1.)

            #print("Test time: {0:.4f}s".format(stop_time - start_time))
        ## if you are using older version of PyTorch, torch.no_grad() may not be supported
        # ISource, INoisy = Variable(ISource.cuda(),volatile=True), Variable(INoisy.cuda(),volatile=True)
        # Out = torch.clamp(INoisy-model(INoisy), 0., 1.)
        psnr = batch_PSNR(Out, ISource, 1.)
        ssim = batch_SSIM(Out, ISource, 1.)
        #ssim = calculate_ssim(Out, ISource)
        psnr_test += psnr
        ssim_test += ssim
        print("%s PSNR %f SSIM %f" % (f, psnr, ssim))

        noise_psnr = batch_PSNR(INoisy, ISource, 1.)
        noise_ssim = batch_SSIM(INoisy, ISource, 1.)
        noise_psnr_test += noise_psnr
        noise_ssim_test += noise_ssim


        # Tensor to Image.
        clean_img = utils.make_grid(ISource.data, nrow=8, normalize=True, scale_each=True)
        denoising_img = utils.make_grid(Out.data, nrow=8, normalize=True, scale_each=True)
        nosie_img = utils.make_grid(INoisy.data, nrow=8, normalize=True, scale_each=True)

        # Image Plot.
        # fig = plt.figure()
        # rows = 1
        # cols = 2
        #
        # # ax1 = fig.add_subplot(rows, cols, 1)
        # # ax1.imshow(np.transpose(clean_img.cpu(), (1,2,0)), cmap="gray")
        # # ax1.set_title('clean image')
        # #
        # # ax2 = fig.add_subplot(rows, cols, 2)
        # # ax2.imshow(np.transpose(denoising_img.cpu(), (1,2,0)), cmap="gray")
        # # ax2.set_title('denoising image')
        #
        # plt.show()

        result_img = torch.clamp(denoising_img * 255, 0, 255)
        result_img = np.uint8(result_img.cpu())
        a = np.transpose(result_img, (1,2,0))
        # imwrite('./fig_result/WDECdenoising/' + f, a)

        noise_img = torch.clamp(nosie_img * 255, 0, 255)
        noise_img = np.uint8(noise_img.cpu())
        #imwrite('./fig_result/WDECdenoising/noise/' + f, np.transpose(noise_img, (1, 2, 0)))

    psnr_test /= len(files_source)
    ssim_test /= len(files_source)

    noise_psnr_test /= len(files_source)
    noise_ssim_test /= len(files_source)

    del (test_results['runtime'][0])
    ave_runtime = sum(test_results['runtime']) / len(test_results['runtime']) / 1000.0

    print("\nPSNR on test data %f; SSIM: %f" % (psnr_test, ssim_test))
    #print("\nnoise_______PSNR: %f; SSIM: %f" % (noise_psnr_test, noise_ssim_test))
    print('------> Average runtime is : {:.6f} seconds'.format(ave_runtime))

if __name__ == "__main__":
    main()
