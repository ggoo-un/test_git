import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models import *
#from dataset import prepare_data, Dataset
from utils import *
import utilspackage
from utilspackage.loader import  get_training_data,get_validation_data

from matplotlib import pyplot as plt
from torchvision.transforms import ToPILImage
import torchvision.transforms as transforms

from imageio import imwrite
import PIL
from PIL import Image
from PIL import ImageFilter
from datetime import datetime

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="WDEN")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=16, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=16, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=55, help="Number of training epochs")
parser.add_argument("--resume_epochs", type=int, default=33, help="Number of training epochs When training resume")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--patchsize", type=int, default=128, help='patch size of image')
parser.add_argument('--train_dir', type=str, default='data/sidd/train', help='dir of train data')
parser.add_argument('--val_dir', type=str, default='data/sidd/val', help='dir of train data')
parser.add_argument('--train_workers', type=int, default=8, help='train_dataloader workers')
parser.add_argument('--eval_workers', type=int, default=6, help='eval_dataloader workers')
opt = parser.parse_args()


def main():
    # Load dataset
    print('Loading dataset ...\n')

    img_options_train = {'patch_size': opt.patchsize}
    train_dataset = get_training_data(opt.train_dir, img_options_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batchSize, shuffle=True,
                              num_workers=opt.train_workers, pin_memory=True, drop_last=False)

    val_dataset = get_validation_data(opt.val_dir)
    val_loader = DataLoader(dataset=val_dataset, batch_size=opt.batchSize, shuffle=False,
                            num_workers=opt.eval_workers, pin_memory=False, drop_last=False)

    len_trainset = train_dataset.__len__()
    len_valset = val_dataset.__len__()
    print("Sizeof training set: ", len_trainset, ", sizeof validation set: ", len_valset)

    # Build model
    net = WDENet(channels=3)
    net.apply(weights_init_kaiming)
    criterion = nn.L1Loss()
    model = net.cuda()

    ## if you want to train resuming
    load = torch.load(os.path.join(opt.outf, 'WDEC25_33_39.2994.pth'))
    model.load_state_dict(load)

    criterion.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # training
    # writer = SummaryWriter(opt.outf)
    #step = 0

    start_time = datetime.now()
    print('Training Start!!')
    print(start_time)

    for epoch in range(opt.resume_epochs, opt.epochs):
        if epoch < opt.milestone:
            current_lr = opt.lr # 1e-4
        else:
            current_lr = opt.lr / 10. # 1e-5

        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)

        # train
        for i, data in enumerate(train_loader, 0):
            # training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            #img_train = data
            target = data[0].cuda()
            input_ = data[1].cuda()

            #noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=opt.noiseL/255.)
            #imgn_train = img_train + noise

            target, input_ = Variable(target), Variable(input_)

            restored = model(input_)
            restored = torch.clamp(restored, 0, 1)

            loss = criterion(restored, target)

            loss.backward()
            optimizer.step()

            # results
            model.eval()
            restored = model(input_)
            restored = torch.clamp(restored, 0., 1.)
            psnr_train = batch_PSNR(restored, target, 1.)
            # i%100 == 0 -> each 100 epochs, print loss and psnr.
            #print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %(epoch + 1, i + 1, len(train_loader), loss.item(), psnr_train))
            if i % 500 == 0 :
                print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" % (epoch+1, i+1, len(train_loader), loss.item(), psnr_train))
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            # if step % 10 == 0:
            #     # Log the scalar values
            #     writer.add_scalar('loss', loss.item(), step)
            #     writer.add_scalar('PSNR on training data', psnr_train, step)
            # step += 1


        # validate
        #for k in range(len(dataset_val)):
        with torch.no_grad():
            model.eval()
            psnr_val_rgb = []
            for ii, data_val in enumerate((val_loader), 0):
                target = data_val[0].cuda()
                input_ = data_val[1].cuda()
                filenames = data_val[2]
                target, input_ = Variable(target.cuda()), Variable(input_.cuda())
                with torch.cuda.amp.autocast():
                    restored = model(input_)
                restored = torch.clamp(restored, 0., 1.)
                psnr_val_rgb.append(utilspackage.batch_PSNR(restored, target, False).item())

            psnr_val_rgb = sum(psnr_val_rgb) / len_valset

        print("[epoch %d] PSNR_val: %.4f\n" % (epoch+1, psnr_val_rgb))
        midtime = datetime.now() - start_time
        print(midtime)
        # writer.add_scalar('PSNR on validation data', psnr_val, epoch)

        # log the images
        # out_train = torch.clamp(imgn_train-model(imgn_train), 0., 1.)
        Img = utils.make_grid(target[0].data, nrow=8, normalize=True, scale_each=True)
        Imgn = utils.make_grid(input_[0].data, nrow=8, normalize=True, scale_each=True)
        Irecon = utils.make_grid(restored[0].data, nrow=8, normalize=True, scale_each=True)
        # writer.add_image('clean image', Img, epoch)
        # writer.add_image('noisy image', Imgn, epoch)
        # writer.add_image('reconstructed image', Irecon, epoch)

        # Compare clean, noisy, denoising image
        fig = plt.figure()
        fig.suptitle(epoch + 1)
        rows = 1
        cols = 3

        ax1 = fig.add_subplot(rows, cols, 1)
        # tensor는 cuda에서 처리하지 못하기 때문에 .cpu()로 보내줌.
        ax1.imshow(np.transpose(Img.cpu(), (1,2,0)))
        ax1.set_title('clean image')

        ax2 = fig.add_subplot(rows, cols, 2)
        ax2.imshow(np.transpose(Imgn.cpu(), (1,2,0)))
        ax2.set_title('noisy image')

        ax3 = fig.add_subplot(rows, cols, 3)
        ax3.imshow(np.transpose(Irecon.cpu(), (1, 2, 0)))
        ax3.set_title('denoising image')

        result_img = torch.clamp(Irecon * 255, 0, 255)
        result_img = np.uint8(result_img.cpu())


        # plt.savefig('./fig_result/epoch_{:d}.png'.format(epoch + 1))
        plt.show()

        imwrite('./fig_result/denoising/result/result.png', np.transpose(result_img, (1, 2, 0)))

        # torch.save(model.state_dict(), os.path.join(opt.outf, 'WDEC_real.pth'))
        # save model
        # torch.save(model.state_dict(), os.path.join(opt.outf, 'UsingIQRnNoiseblock_Dualnet_25.pth'))
        # nl(noise level)25 => 30.6000 nl15 => 32.8000 nl50 => 27.3000
        if psnr_val_rgb >= 39.0000:
            torch.save(model.state_dict(), os.path.join(opt.outf, 'WDEC_real_' + str(epoch + 1) + "_" + str(round(psnr_val_rgb, 4)) + '.pth'))

    end_time = datetime.now()
    print('Training Finished!!')
    print(end_time)

if __name__ == "__main__":

    main()
