import torch
import torch.nn as nn
from utils import *
from matplotlib import pyplot as plt

class WDENet(nn.Module):
    def __init__(self, channels):
        super(WDENet, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64

        n_head = []
        n_head.append(
            nn.Conv2d(in_channels=4, out_channels=features, kernel_size=kernel_size, padding=padding,
                      bias=False))
        self.head = nn.Sequential(*n_head)

        dw_l1 = []
        for _ in range(3):
            dw_l1.append(BBlock(features, features, 3, bn=True))

        dw_l2 = [BBlock(features * 4, features * 4, 3, bn=True)]
        for _ in range(3):
            dw_l2.append(BBlock(features * 4, features * 4, 3, bn=True))

        dw_l3 = [BBlock(features * 16, features * 4, 3, bn=True)]
        for _ in range(3 * 2):
            dw_l3.append(BBlock(features * 4, features * 4, 3, bn=True))
        dw_l3.append(BBlock(features * 4, features * 16, 3, bn=True))

        iw_l2 = []
        for _ in range(3):
            iw_l2.append(BBlock(features * 4, features * 4, 3, bn=True))
        iw_l2.append(BBlock(features * 4, features * 4, 3, bn=True))

        iw_l1 = []
        for _ in range(3):
            iw_l1.append((BBlock(features, features, 3, bn=True)))

        m_tail = [BBlock(features, 4, 3)]

        self.dw_l2 = nn.Sequential(*dw_l2)
        self.dw_l1 = nn.Sequential(*dw_l1)
        self.dw_l3 = nn.Sequential(*dw_l3)
        self.iw_l2 = nn.Sequential(*iw_l2)
        self.iw_l1 = nn.Sequential(*iw_l1)
        self.tail = nn.Sequential(*m_tail)

        ####### 25, 50
        # layers1 = []
        # layers1.append(
        #     nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
        #               bias=False))
        # layers1.append(nn.ReLU(inplace=True))
        # for _ in range(4):
        #     layers1.append(
        #         nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
        #                   bias=False))
        #     layers1.append(nn.ReLU(inplace=True))
        #
        # layers1.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding,
        #                          bias=False))
        #
        # self.conv_2 = nn.Conv2d(in_channels=channels * 2, out_channels=features, kernel_size=kernel_size,
        #                         padding=padding, bias=False)
        #
        # self.detail_b1 = nn.Sequential(*layers1)
        ######

        layers2 = []
        layers2.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                                 bias=False))
        layers2.append(nn.ReLU(inplace=True))
        for _ in range(15):
            layers2.append(
                nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers2.append(nn.BatchNorm2d(features))
            layers2.append(nn.ReLU(inplace=True))
        layers2.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                                 bias=False))

        layers3 = []
        layers3.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                                 bias=False))
        layers3.append(nn.ReLU(inplace=True))

        for _ in range(6):
            layers3.append(
                nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers3.append(nn.ReLU(inplace=True))

        layers3.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                                 bias=False))

        block = []
        for _ in range(3):
            block.append(
                nn.Conv2d(in_channels=features * 2, out_channels=features * 2, kernel_size=kernel_size, padding=padding,
                          bias=False))
            block.append(nn.BatchNorm2d(features * 2))
            block.append(nn.ReLU(inplace=True))
        block.append(
            nn.Conv2d(in_channels=features * 2, out_channels=channels, kernel_size=kernel_size, padding=padding,
                      bias=False))


        self.step_1 = nn.Sequential(*layers2)
        self.add = nn.Sequential(*block)
        self.step_2 = nn.Sequential(*layers3)

    def forward(self, x):
        residual = x
        h_data = self.head(dwt_init(x))

        w_data1 = self.dw_l1(h_data)
        w_data2 = self.dw_l2(dwt_init(w_data1))
        # x3 = self.d_l2(self.DWT(x2))
        iw_data = iwt_init(self.dw_l3(dwt_init(w_data2))) + w_data2
        iw_data = iwt_init(self.iw_l2(iw_data)) + w_data1

        # x = self.i_l0(x) + x0
        w_x = iwt_init(self.tail(self.iw_l1(iw_data))) + residual

        # fig = plt.figure()
        # rows = 1
        # cols = 2
        #
        # ax1 = fig.add_subplot(rows, cols, 1)
        # ax1.imshow(np.transpose(x[0,:,:,:].cpu(), (1, 2, 0)), cmap="gray")
        # ax1.set_title('clean image')
        #
        # ax2 = fig.add_subplot(rows, cols, 2)
        # ax2.imshow(np.transpose(w_x[0,:,:,:].cpu(), (1, 2, 0)), cmap="gray")
        # ax2.set_title('feature image')
        #
        # plt.show()

        out0 = self.step_2(w_x)
        out2 = self.step_1(residual)

        # noise 15, 25
        data_cat2 = torch.cat((out0, residual - out2), 1)

        # noise 50
        # data_cat2 = torch.cat((out0, out2 + residual), 1)

        out3 = self.add(data_cat2)

        out = out3 + residual

        return out