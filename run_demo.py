from __future__ import print_function, absolute_import

import os
import sys
import argparse
from time import time
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
from PIL import Image

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision

from models import pvcNet
import utils
from utils.imutils import show_voxel, show_joints3D


def img_crop(image_tensor, center, scale):
    return utils.transforms.crop(image_tensor, center, scale, [256, 256])


def transf_pred(pred_coord, center, scale):
    lm_pred = utils.transforms.transform_preds(pred_coord, center, scale, [256, 256], 256)

    lm_pred[:, 2] = -lm_pred[:, 2]

    z_mean = torch.mean(lm_pred[:, 2])
    lm_pred[:, 2] -= z_mean

    return lm_pred


def main(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    if args.gpus == '':
        is_cuda = False
        print('Run in CPU mode.')
    else:
        is_cuda = True
        cudnn.benchmark = True

    # 68 points plus eye and mouth center
    nParts = 71
    # create model
    print("==> creating model: stacks={}, blocks={}, z-res={}".format(args.stacks, args.blocks, args.depth_res))
    model = pvcNet(args.stacks, args.blocks, args.depth_res, nParts,
                   resume_p2v2c=args.resume_p2v2c, is_cuda=is_cuda)

    imgDir = args.imgDir
    lmDir = args.lmDir
    outDir = args.outDir

    model.resume_from_checkpoint()

    model.eval()

    imgPathList = [imgDir+name for name in os.listdir(imgDir) if name.endswith('.jpg')]

    # link for facial points
    skeletons = [[i, i + 1] for i in range(16)] + \
                 [[i, i + 1] for i in range(17, 21)] + \
                 [[i, i + 1] for i in range(22, 26)] + \
                 [[i, i + 1] for i in range(36, 41)] + [[41, 36]] + \
                 [[i, i + 1] for i in range(42, 47)] + [[47, 42]] + \
                 [[i, i + 1] for i in range(27, 30)] + \
                 [[i, i + 1] for i in range(31, 35)] + \
                 [[i, i + 1] for i in range(48, 59)] + [[59, 48]] + \
                 [[i, i + 1] for i in range(60, 67)] + [[67, 60]]

    nme = []
    for i, path in enumerate(imgPathList):

        image = Image.open(path).convert('RGB')
        imgId = path.split('/')[-1][:-4]

        if lmDir is not None:
            ptsPath = lmDir + imgId + '.pts'
            lm_gt = utils.evaluation.readPts(ptsPath)

            center, scale, bbox = utils.evaluation.boundingbox(lm_gt)
            scale *= 1.25

            l = center[0] - scale*200/2.
            u = center[1] - scale*200/2.
            w = scale*200
            h = scale*200

            bbox = [l, u, l + w, u + h]

            lm_gt = torch.Tensor(lm_gt)
        else:
            center = np.array(image.size)/2
            scale = 1

        image_tensor = torchvision.transforms.ToTensor()(image)
        # input image with size of 256x256
        input_tensor = img_crop(image_tensor, center, scale)
        if is_cuda:
            input_tensor = input_tensor.cuda()

        timeStart = time.time()
        pred_voxel, pred_coord = model.landmarkDetection(input_tensor.unsqueeze(0))
        timeElapse = time.time() - timeStart

        pred_coord = pred_coord.data[0:68]
        if is_cuda:
            pred_coord = pred_coord.cpu()

        if args.verbose:
            utils.imutils.show_joints(input_tensor, pred_coord[:,0:2], show_idx=False, pairs=skeletons,
                                      ax=plt.subplot(221))

            # visualize the last volume
            pred_heatmap3d = [pred_voxel[-1].data[0].cpu().numpy()]
            show_voxel(pred_heatmap3d, ax=plt.subplot(222, projection='3d'))

            show_joints3D(pred_coord.numpy(), pairs=skeletons, ax=plt.subplot(223, projection='3d'))
            plt.show()

        if lmDir is not None:
            lm_pred = transf_pred(pred_coord, center, scale)
            err = utils.evaluation.p2pNormMeanError(lm_pred.unsqueeze(0), lm_gt.unsqueeze(0), [36, 45], z_zero_mean=True)[0]
            nme.append(err)

        if outDir is not None:
            # write 3D coordinates to file
            text_file = open(outDir + imgId + '.csv', "w")
            for idx in range(68):
                text_file.write("{},{},{}\n".format(lm_pred[idx, 0], lm_pred[idx, 1], lm_pred[idx, 2]))
            text_file.close()

        sys.stdout.write('\r')
        sys.stdout.write('{}/{} Done; Elapse {:.0f}ms'.format(i + 1, len(imgPathList), timeElapse*1000))
        sys.stdout.flush()

    if len(nme) > 0:
        print('\nGround Truth Error (GTE): {:.4f}'.format(np.mean(np.array(nme))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Joint Voxel and Coordinate Regression')

    parser.add_argument('-s', '--stacks', default=4, type=int, metavar='N',
                        help='number of hourglasses to stack')
    parser.add_argument('-b', '--blocks', default=1, type=int, metavar='N',
                        help='number of residual modules at each location in the hourglass')
    parser.add_argument('--depth_res', default=[1, 2, 4, 64], type=int, nargs="*",
                        help='Resolution of depth for the output of the corresponding hourglass')
    parser.add_argument('--resume_p2v', default='', type=str,
                        help='path to the model of voxel regression subnetwork')
    parser.add_argument('--resume_v2c', default='', type=str,
                        help='path to the model of coordinate regression subnetwork')
    parser.add_argument('--resume_p2v2c', default='checkpoint/model_p2v2c_300wLP.pth.tar', type=str,
                        help='path to the pre-trained model')
    parser.add_argument('--gpus', default='0', type=str, help='set gpu IDs')
    parser.add_argument('--imgDir', default='./imgs/', type=str, help='path to test images')
    parser.add_argument('--lmDir', default='./imgs/', type=str, help='path to ground-truth .pts files')
    parser.add_argument('--outDir', default='./output/', type=str, help='path for saving prediction results')
    parser.add_argument('--verbose', default=False, action='store_true', help='enable visualization')

    main(parser.parse_args())
