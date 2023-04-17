import os
import time
import torch
#######
import numpy as np
from torch.autograd import Variable
from FFDNet.models import FFDNet
import torch.nn as nn
from matplotlib import pyplot as plt


#############
# load a FFDNet model for denoising
model_fn = 'FFDNet/net_rgb.pth'
# Absolute path to model file
model_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)), model_fn)
# Create model
net = FFDNet(num_input_channels=3)
# Load saved weights
state_dict = torch.load(model_fn)
device_ids = [0]
modelFFD = nn.DataParallel(net, device_ids=device_ids).cuda()
modelFFD.load_state_dict(state_dict)
# Sets the model in evaluation mode (e.g. it removes BN)
modelFFD.eval()
##############


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

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


#######
def FFDNet_AddDe(imorig, sigma):
    """
    noise addition then denoising
    """
    dtype = torch.cuda.FloatTensor

    # Add noise
    imnoisy = imorig + torch.randn(imorig.size()[0], 3, imorig.size()[2], imorig.size()[3])*sigma
    # Test mode
    with torch.no_grad():
        imnoisy = Variable(imnoisy.type(dtype))
        nsigma = Variable(torch.FloatTensor([sigma]).type(dtype))

        # Estimate noise and subtract it to the input image
        nsigma_repeat = nsigma.repeat(imorig.size()[0])
        im_noise_estim = modelFFD(imnoisy, nsigma_repeat)
        outim = torch.clamp(imnoisy - im_noise_estim, 0., 1.)

    return outim


def test_epoch_AddDe(model, loader, print_freq=1, is_test=True, noise_sigma=5/255.):
    batch_time = AverageMeter()
    error = AverageMeter()

    end = time.time()
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            # ###################
            # input: [0 1] tensor N*C*H*W;
            input_AddDe = FFDNet_AddDe(input, noise_sigma)
            ####
            # Create vaiables
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            output_AddDe = model(input_AddDe)

            # measure accuracy and record loss
            batch_size = target.size(0)
            _, pred = output.data.cpu().topk(1, dim=1)
            _, pred_AddDe = output_AddDe.data.cpu().topk(1, dim=1)

            error.update(torch.ne(pred.squeeze(), pred_AddDe.squeeze()).float().sum().item() / batch_size, batch_size)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print stats
            if batch_idx % print_freq == 0:
                res = '\t'.join([
                    'Test' if is_test else 'Valid',
                    'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                    'Error %.4f (%.4f)' % (error.val, error.avg),
                ])
                print(res)

    # Return summary statistics
    return batch_time.avg, error.avg
