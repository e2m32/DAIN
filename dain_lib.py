import os
from torch.autograd import Variable
import torch
from torch.nn import functional as f
import numpy
import networks
from my_args import args
from scipy.misc import imread, imsave
import gc
# import time
# import math
# import random
# from AverageMeter import *

# to speed up the processing
torch.backends.cudnn.benchmark = True

TEMP_PNG = 'temp.png'


class Dain(object):
    def __init__(self, trained_weights='./model_weights/best.pth'):
        # Check for trained weights
        if not os.path.exists(trained_weights):
            error_message = str(trained_weights) + " trained weights could not be found"
            print('*' * (len(error_message) + 10))
            print("**** " + error_message + " ****")
            print('*' * (len(error_message) + 10))
            raise FileNotFoundError

        args.SAVED_MODEL = trained_weights
        print("The testing model weight is: " + args.SAVED_MODEL)

        # Setup Cuda for modeling
        self.use_cuda = args.use_cuda
        self.save_which = args.save_which
        self.dtype = args.dtype

        self.model = networks.__dict__[args.netName]\
                                      (
                                        channel=args.channels,
                                        filter_size=args.filter_size,
                                        timestep=args.time_step,
                                        training=False
                                      )

        if not self.use_cuda:
            pretrained_dict = torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage)
            # self.model.load_state_dict(torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage))
        else:
            pretrained_dict = torch.load(args.SAVED_MODEL)
            # self.model.load_state_dict(torch.load(args.SAVED_MODEL))

        if self.use_cuda:
            self.model = self.model.cuda()
        model_dict = self.model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.model.load_state_dict(model_dict)
        # 4. release the pretrained dict for saving memory
        del pretrained_dict
        del model_dict

    def dain_interpolate(self, image1, image2):
        """
        Run DAIN processing
        :param image1: First image to compare
        :param image2: Compare this to first image
        :return: image interpolated between image1 and image2
        """
        # deploy model
        self.model = self.model.eval()

        # interp_error = AverageMeter()
        # tot_timer = AverageMeter()
        # proc_timer = AverageMeter()
        # end = time.time()

        X0 = torch.from_numpy(numpy.transpose(image1, (2, 0, 1)).astype("float32") / 255.0).type(self.dtype)
        X1 = torch.from_numpy(numpy.transpose(image2, (2, 0, 1)).astype("float32") / 255.0).type(self.dtype)
        y_ = torch.FloatTensor()

        assert(X0.size(1) == X1.size(1))
        assert(X0.size(2) == X1.size(2))
        int_width = X0.size(2)
        int_height = X0.size(1)
        channel = X0.size(0)
        if not channel == 3:
            raise Exception(image1+' has too many channels, cannot process this image.')

        if int_width != ((int_width >> 7) << 7):
            intWidth_pad = (((int_width >> 7) + 1) << 7)  # more than necessary
            intPaddingLeft = int((intWidth_pad - int_width) / 2)
            intPaddingRight = intWidth_pad - int_width - intPaddingLeft
        else:
            intWidth_pad = int_width
            intPaddingLeft = 32
            intPaddingRight= 32

        if int_height != ((int_height >> 7) << 7):
            intHeight_pad = (((int_height >> 7) + 1) << 7)  # more than necessary
            intPaddingTop = int((intHeight_pad - int_height) / 2)
            intPaddingBottom = intHeight_pad - int_height - intPaddingTop
        else:
            intHeight_pad = int_height
            intPaddingTop = 32
            intPaddingBottom = 32

        pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight, intPaddingTop, intPaddingBottom])

        torch.set_grad_enabled(False)
        X0 = Variable(torch.unsqueeze(X0, 0))
        X1 = Variable(torch.unsqueeze(X1, 0))
        X0 = pader(X0)
        X1 = pader(X1)

        # print("***I was able to do padder stuff.***")
        # print("{}mb of GPU memory in use".format(torch.cuda.memory_allocated(device=None) / 1000000))

        if self.use_cuda:
            X0 = X0.cuda()
            X1 = X1.cuda()
        # proc_end = time.time()
        y_s, offset, filter = self.model(torch.stack((X0, X1), dim=0))
        y_ = y_s[self.save_which]

        # proc_timer.update(time.time() - proc_end)
        # tot_timer.update(time.time() - end)
        # end = time.time()
        # message = "current image process time \t " + str(time.time() - proc_end)+"s"
        # print('*' * (len(message) + 10))
        # print("**** " + message + " ****")
        # print('*' * (len(message) + 10))

        if self.use_cuda:
            X0 = X0.data.cpu().numpy()
            y_ = y_.data.cpu().numpy()
            offset = [offset_i.data.cpu().numpy() for offset_i in offset]
            filter = [filter_i.data.cpu().numpy() for filter_i in filter] if filter[0] is not None else None
            X1 = X1.data.cpu().numpy()
        else:
            X0 = X0.data.numpy()
            y_ = y_.data.numpy()
            offset = [offset_i.data.numpy() for offset_i in offset]
            filter = [filter_i.data.numpy() for filter_i in filter]
            X1 = X1.data.numpy()

        X0 = numpy.transpose(255.0 *
                             X0.clip(0, 1.0)[
                               0,
                               :,
                               intPaddingTop:intPaddingTop + int_height,
                               intPaddingLeft:intPaddingLeft + int_width
                             ],
                             (1, 2, 0))
        y_ = numpy.transpose(255.0 *
                             y_.clip(0, 1.0)
                             [
                               0,
                               :,
                               intPaddingTop:intPaddingTop + int_height,
                               intPaddingLeft:intPaddingLeft + int_width
                             ],
                             (1, 2, 0))
        offset = [numpy.transpose
                  (
                    offset_i
                    [
                        0,
                        :,
                        intPaddingTop:intPaddingTop + int_height,
                        intPaddingLeft:intPaddingLeft + int_width],
                        (1, 2, 0)
                  ) for offset_i in offset]
        filter = [numpy.transpose
                  (
                    filter_i
                    [
                        0,
                        :,
                        intPaddingTop:intPaddingTop + int_height,
                        intPaddingLeft: intPaddingLeft + int_width
                    ],
                    (1, 2, 0)
                  ) for filter_i in filter] if filter is not None else None
        X1 = numpy.transpose(255.0 *
                             X1.clip(0, 1.0)
                             [
                               0,
                               :,
                               intPaddingTop:intPaddingTop + int_height,
                               intPaddingLeft:intPaddingLeft + int_width
                             ],
                             (1, 2, 0))

        imsave(TEMP_PNG, numpy.round(y_).astype(numpy.uint8))
        rec_rgb = imread(TEMP_PNG)

        print("{:2} mb of GPU memory in use".format(torch.cuda.memory_allocated(device=None) / 1000000))

        # clear memory
        del X1
        del filter
        del y_
        del X0
        gc.collect()
        torch.cuda.empty_cache()

        # gt_rgb = imread(gt_path)
        #
        # diff_rgb = 128.0 + rec_rgb - gt_rgb
        # avg_interp_error_abs = numpy.mean(numpy.abs(diff_rgb - 128.0))
        #
        # interp_error.update(avg_interp_error_abs, 1)
        #
        # mse = numpy.mean((diff_rgb - 128.0) ** 2)
        #
        # PIXEL_MAX = 255.0
        # psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
        #
        # print("interpolation error / PSNR : " + str(round(avg_interp_error_abs,4)) + " / " + str(round(psnr,4)))
        # metrics = "The average interpolation error / PSNR for all images are : " + str(round(interp_error.avg, 4))
        # print(metrics)

        return rec_rgb
