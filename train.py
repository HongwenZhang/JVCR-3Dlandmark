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
from utils.misc import adjust_learning_rate
from utils.evaluation import AverageMeter, bboxNormMeanError, p2pNormMeanError
from progress.bar import Bar

from datasets import fa68pt3D


def main(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    if args.gpus == '':
        is_cuda = False
        print('Run in CPU mode.')
    else:
        is_cuda = True
        cudnn.benchmark = True

    # set training and evaluation datasets
    train_loader = torch.utils.data.DataLoader(
        fa68pt3D('data/300wLP/300wLP_anno_tr.json', 'data/300wLP/images', '300wLP', depth_res=args.depth_res,
            nStack=args.stacks, sigma=args.sigma, rot_factor=40, jitter=True),
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        fa68pt3D('data/aflw2000/aflw2000_3D_anno_vd.json', 'data/aflw2000/images', 'aflw2000', depth_res=args.depth_res,
            nStack=args.stacks, sigma=args.sigma, train=False),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # 68 points plus eye and mouth center
    nParts = 71
    # create model
    print("==> creating model: stacks={}, blocks={}, z-res={}".format(args.stacks, args.blocks, args.depth_res))
    model = pvcNet(args.stacks, args.blocks, args.depth_res, nParts,
                   resume_p2v2c=args.resume_p2v2c, is_cuda=is_cuda)

    if any([args.resume_p2v, args.resume_v2c, args.resume_p2v2c]):
        print('load checkpoint')
        model.resume_from_checkpoint()

    # set optimizer
    print('using ADAM optimizer.')
    optimizer_G = torch.optim.Adam(model.pix2vox.parameters())
    optimizer_P = torch.optim.Adam(model.vox2coord.parameters())

    # set loss criterion
    criterion_vox = torch.nn.MSELoss(size_average=True).cuda()
    criterion_coord = torch.nn.MSELoss(size_average=True).cuda()

    if args.evaluate:
        print('\nEvaluation only')
        mode = 'evaluate'
        run(model, val_loader, mode, criterion_vox, criterion_coord, optimizer_G, optimizer_P)
        return

    lr = args.lr
    for epoch in range(args.start_epoch, args.epochs):
        lr_new = adjust_learning_rate(optimizer_G, epoch, lr, args.schedule, args.gamma)
        lr_new = adjust_learning_rate(optimizer_P, epoch, lr, args.schedule, args.gamma)
        lr = lr_new
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

        # train for one epoch
        mode = 'pre_train' if epoch < args.pretr_epochs else 'train'
        print(mode+'ing...')
        run(model, train_loader, mode, criterion_vox, criterion_coord, optimizer_G, optimizer_P)

        # evaluation
        mode = 'evaluate'
        _, nme_results = run(model, val_loader, mode, criterion_vox, criterion_coord, optimizer_G,
                                       optimizer_P)

        model.save_to_checkpoint(nme_results, args.checkpoint, snapshot=args.num_snapshot)


def run(model, data_loader, mode, criterion_vox, criterion_coord, optimizer_G, optimizer_P):
    # self.epoch += 1
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses_vox = AverageMeter()
    losses_coord = AverageMeter()
    errs = AverageMeter()

    log_key = ['losses_vox', 'losses_coord', 'errs']
    log_info = dict.fromkeys(log_key)

    # normalized mean error results
    dataset_len = data_loader.dataset.__len__()
    nme_results = torch.Tensor(dataset_len, 1)

    def data2variable(inputs, target, meta):
        if mode in ['pre_train', 'train']:
            input_var = torch.autograd.Variable(inputs.cuda())
            target_var = [torch.autograd.Variable(target[i].cuda(async=True)) for i in range(len(target))]
            coord_var = torch.autograd.Variable(meta['tpts_inp'].cuda(async=True))
        else:
            input_var = torch.autograd.Variable(inputs.cuda(), volatile=True)
            target_var = [torch.autograd.Variable(target[i].cuda(async=True), volatile=True) for i in
                          range(len(target))]
            coord_var = torch.autograd.Variable(meta['tpts_inp'].cuda(async=True), volatile=True)

        return input_var, target_var, coord_var

    # switch mode
    if mode in ['pre_train', 'train']:
        model.pix2vox.train()
        model.vox2coord.train()
    else:
        model.pix2vox.eval()
        model.vox2coord.eval()

    data_num = 0
    data_length = len(data_loader.dataset)

    # measure time
    end = time.time()

    bar = Bar('Processing', max=len(data_loader))
    for i, (inputs, target, meta) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input_var, target_var, coord_var = data2variable(inputs, target, meta)

        # run forward
        pred_vox, _, pred_coord = model.forward(input_var)

        loss_vox = criterion_vox(pred_vox[0], target_var[0])
        for j in range(1, len(pred_vox)):
            loss_vox += criterion_vox(pred_vox[j], target_var[j])

        if mode is 'pre_train':
            # pre-train the coordinate regressor
            voxel_gt = target_var[-1].unsqueeze(1)
            _, _, v2c_out = model.vox2coord(voxel_gt)

            v2c_out = v2c_out.view(inputs.shape[0], -1, 3)
            loss_coord = criterion_coord(v2c_out, coord_var / 255)

            optimizer_G.zero_grad()
            optimizer_P.zero_grad()
            loss_vox.backward()
            loss_coord.backward()
            optimizer_G.step()
            optimizer_P.step()

        elif mode is 'train':
            loss_coord = criterion_coord(pred_coord, coord_var / 255)
            loss_p2v2c = loss_coord + 0.1 * loss_vox

            optimizer_G.zero_grad()
            optimizer_P.zero_grad()
            loss_p2v2c.backward()
            optimizer_G.step()
            optimizer_P.step()

        else:
            loss_coord = criterion_coord(pred_coord, coord_var / 255)

        pred_landmarks = 255 * pred_coord[:, 0:68, :].data
        target_landmarks = meta['tpts_inp'][:, 0:68, :]
        box_nme = bboxNormMeanError(pred_landmarks, target_landmarks)
        # box_nme = p2pNormMeanError(pred_landmarks, target_landmarks, args.norm_idx)
        box_nme = np.array(box_nme)

        for n in range(len(meta['index'])):
            nme_results[meta['index'][n]] = box_nme[n]

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        data_num += len(input_var)

        # measure nme and record loss
        losses_vox.update(loss_vox.data[0], inputs.size(0))
        losses_coord.update(loss_coord.data[0], inputs.size(0))
        errs.update(np.mean(box_nme), inputs.size(0))

        log_info['losses_vox'] = losses_vox.avg
        log_info['losses_coord'] = losses_coord.avg
        log_info['errs'] = errs.avg

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | LOSS {loss} | NME: {nme: .4f}'.format(
            batch=data_num,
            size=data_length,
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss='vox: {:.4f}; coord: {:.4f}'.format(loss_vox.data[0], loss_coord.data[0]),
            nme=errs.avg
        )
        bar.next()

    bar.finish()

    if mode is 'evaluate':
        model.current_acc = -errs.avg
        print('Performance(NME) current: {}, best:{}'.format(-model.current_acc, -model.best_acc))

    log_info['errs'] = -errs.avg

    return log_info, nme_results


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
    parser.add_argument('--resume_p2v2c', default='', type=str,
                        help='path to the pre-trained model')
    parser.add_argument('--gpus', default='0', type=str, help='set gpu IDs')
    # Training strategy
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--pretr_epochs', default=15, type=int, metavar='N',
                        help='Number of epochs for pre-training the network')
    parser.add_argument('--train-batch', default=20, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=50, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=2.5e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--schedule', type=int, nargs='+', default=[10, 20, 30],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--sigma', type=float, default=1,
                        help='Groundtruth Gaussian sigma.')
    parser.add_argument('--num_snapshot', default=5, type=int, metavar='N',
                        help='Frequency for saving checkpoints')
    # Miscs
    parser.add_argument('-c', '--checkpoint', default='checkpoint/300wLP', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')

    main(parser.parse_args())
