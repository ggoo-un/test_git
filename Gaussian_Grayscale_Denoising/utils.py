import math
import torch
import torch.nn as nn
import numpy as np
import skimage
from skimage.measure.simple_metrics import compare_psnr
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import cv2
import torchvision.utils as utils
from matplotlib import pyplot as plt

import torch.nn.functional as F

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant_(m.bias.data, 0.0)

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        #PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
        PSNR += skimage.metrics.peak_signal_noise_ratio(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return (PSNR/Img.shape[0])

def batch_PSNR_bm3d(img, imclean, data_range):
    Img = img.astype(np.float32)
    Iclean = imclean.astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

def batch_SSIM(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    SSIM = 0
    for i in range(Img.shape[0]):
        SSIM += structural_similarity(Iclean[i,:,:,:], Img[i,:,:,:], win_size = 1,data_range = data_range, multichannel=True, gaussian_weights=True, use_sample_covariance=False)
    return (SSIM/Img.shape[0])


# --------------------------------------------
# SSIM
# --------------------------------------------
def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1 = img1.squeeze().cpu().numpy().astype(np.float32)
    img2 = img2.squeeze().cpu().numpy().astype(np.float32)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def data_augmentation(image, mode):
    out = np.transpose(image, (1,2,0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2,0,1))

def variable_to_cv2_image(varim):
    """
    Norm Variable -> Cv2
    """
    nchannels = varim.size()[1]
    if nchannels == 1:
        res = (varim.data.cpu().numpy()[0, 0, :] * 255.).clip(0, 255).astype(np.uint8)
    elif nchannels == 3:
        res = varim.data.cpu().numpy()[0]
        res = cv2.cvtColor(res.transpose(2, 1, 0), cv2.COLOR_RGB2BGR)
        res = (res*255.).clip(0, 255).astype(np.uint8)
    else:
        raise Exception('Number of color channels not supported')
    return res

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def edge_init(x):
    in_batch, in_channel, in_height, in_width = x.size()
    for i in range(in_batch):
        n_x = x[i,:,:,:]
        n_x = n_x.unsqueeze(0)
        conv1 = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        torch.Size([1, 1, 3, 3])
        w1 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype='float32')
        w1 = w1.reshape(1, 1, 3, 3)
        #w1 = np.repeat(w1, 16, axis=0)
        w1 = torch.Tensor(w1).cuda()
        conv1.weight = nn.Parameter(w1)

        conv2 = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        w2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype='float32')
        w2 = w2.reshape(1, 1, 3, 3)
        #w2 = np.repeat(w2, 16, axis=0)
        w2 = torch.Tensor(w2).cuda()
        conv2.weight = nn.Parameter(w2)

        conv3 = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        w3 = np.array([[-1, 0, 1], [-2, 0, 1], [-1, 0, 1]], dtype='float32')
        w3 = w3.reshape(1, 1, 3, 3)
        #w3 = np.repeat(w3, 16, axis=0)
        w3 = torch.Tensor(w3).cuda()
        conv3.weight = nn.Parameter(w3)

        conv4 = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        w4 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
        #w4 = np.array([[1, 1, -1], [1, -1, 1], [-1, 1, 1]], dtype='float32')
        w4 = w4.reshape(1, 1, 3, 3)
        w4 = torch.Tensor(w4).cuda()
        #w4 = np.repeat(w4, 16, axis=0)
        conv4.weight = nn.Parameter(w4)

        y1 = conv1(n_x)
        y2 = conv2(n_x)
        y3 = conv3(n_x)
        y4 = conv4(n_x)

        # y_1 = utils.make_grid(y1.data, nrow=8, normalize=True, scale_each=True)
        # y_2 = utils.make_grid(y2.data, nrow=8, normalize=True, scale_each=True)
        # y_3 = utils.make_grid(y3.data, nrow=8, normalize=True, scale_each=True)
        # y_4 = utils.make_grid(y4.data, nrow=8, normalize=True, scale_each=True)
        #
        #
        # fig = plt.figure()
        # rows = 2
        # cols = 2
        #
        # ax1 = fig.add_subplot(rows, cols, 1)
        # ax1.imshow(np.transpose(y_1.cpu(), (1, 2, 0)), cmap="gray")
        # ax1.set_title('y1 image')
        #
        # ax2 = fig.add_subplot(rows, cols, 2)
        # ax2.imshow(np.transpose(y_2.cpu(), (1, 2, 0)), cmap="gray")
        # ax2.set_title('y2 image')
        #
        # ax3 = fig.add_subplot(rows, cols, 3)
        # ax3.imshow(np.transpose(y_3.cpu(), (1, 2, 0)), cmap="gray")
        # ax3.set_title('y3 image')
        #
        # ax4 = fig.add_subplot(rows, cols, 4)
        # ax4.imshow(np.transpose(y_4.cpu(), (1, 2, 0)), cmap="gray")
        # ax4.set_title('y4 image')
        #
        # plt.show()

        cat = torch.cat((y1, y2, y3, y4), 1)
        if i==0:
            cat_data = torch.cat((y1, y2, y3, y4), 1)
        else:
            cat_data = torch.cat((cat_data, cat), 0)

    return cat_data

def dwt_init(x):
    h_s01 = len(x[0, 0, 0::2, 0])
    h_s02 = len(x[0, 0, 1::2, 0])
    w_s01 = len(x[0, 0, 0, 0::2])
    w_s02 = len(x[0, 0, 0, 1::2])

    x01 = x[:, :, 0::2, :] / 2
    if h_s01>h_s02:
        x02 = x[:, :, 1::2, :] / 2
        x02 = F.pad(input=x02,pad=(0,0,0,1),mode='reflect') #pad=(左,右,上,下)
    else:
        x02 = x[:, :, 1::2, :] / 2

    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    if w_s01 > w_s02:
        x3 = x01[:, :, :, 1::2]
        x3 = F.pad(input=x3, pad=(0, 1, 0, 0), mode='reflect')
        x4 = x02[:, :, :, 1::2]
        x4 = F.pad(input=x4, pad=(0, 1, 0, 0), mode='reflect')
    else:
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]

    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    if in_height % 2 ==1:
        h_s = -1
    else:
        h_s = 0

    if in_width % 2 == 1:
        w_s = -1
    else:
        w_s = 0
    # print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height + h_s, r * in_width + w_s

    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()
    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4

    if (in_height % 2 == 1) and (in_width % 2 == 1):
        h[:, :, 1::2, 0::2] = (x1 - x2 + x3 - x4)[:, :, :-1, :]
        h[:, :, 0::2, 1::2] = (x1 + x2 - x3 - x4)[:, :, :, :-1]
        h[:, :, 1::2, 1::2] = (x1 + x2 + x3 + x4)[:, :, :-1, :-1]
    else:
        h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4


    return h
# def dwt_init(x):
#     x01 = x[:, :, 0::2, :] / 2
#     x02 = x[:, :, 1::2, :] / 2
#     x1 = x01[:, :, :, 0::2]
#     x2 = x02[:, :, :, 0::2]
#     x3 = x01[:, :, :, 1::2]
#     x4 = x02[:, :, :, 1::2]
#     x_LL = x1 + x2 + x3 + x4
#     x_HL = -x1 - x2 + x3 + x4
#     x_LH = -x1 + x2 - x3 + x4
#     x_HH = x1 - x2 - x3 + x4
#
#     return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)
#
#
# def iwt_init(x):
#     r = 2
#     in_batch, in_channel, in_height, in_width = x.size()
#     # print([in_batch, in_channel, in_height, in_width])
#     out_batch, out_channel, out_height, out_width = in_batch, int(
#         in_channel / (r ** 2)), r * in_height, r * in_width
#     x1 = x[:, 0:out_channel, :, :] / 2
#     x2 = x[:, out_channel:out_channel * 2, :, :] / 2
#     x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
#     x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
#
#     h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()
#
#     h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
#     h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
#     h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
#     h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
#
#     return h

class BBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, padding=1,
        bias=False, bn=False, act=nn.ReLU(True), res_scale=1):

        super(BBlock, self).__init__()
        m = []
        m.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias))
        if bn: m.append(nn.BatchNorm2d(out_channels))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x