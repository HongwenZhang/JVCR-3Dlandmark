import torch
import torch.nn as nn
from hourglass import Bottleneck
from hourglass import HourglassNet
from coordRegressor import coordRegressor

from utils.osutils import mkdir_p, isfile, isdir, join
from utils.misc import save_checkpoint, save_pred


class pvcNet(nn.Module):
    '''H. Zhang et al. Joint Voxel and Coordinate Regression for Accurate 3D Facial Landmark Localization'''
    def __init__(self, stacks, blocks, depth_res, nParts,
                 resume_p2v2c=None, resume_p2v=None, resume_v2c=None, is_cuda=True):
        super(pvcNet, self).__init__()

        self.num_stacks = stacks
        self.num_blocks = blocks
        self.depth_res = depth_res
        self.nParts = nParts

        self.resume = dict()
        self.resume['p2v2c'] = resume_p2v2c
        self.resume['p2v'] = resume_p2v
        self.resume['v2c'] = resume_v2c
        self.is_cuda = is_cuda

        self.epoch = 0
        self.best_acc = 0
        self.current_acc = -0

        if len(set(self.depth_res)) == 1:
            self.c2f = False
        else:
            self.c2f = True

        print('coarse to fine mode: {}'.format(self.c2f))

        pix2vox = HourglassNet(Bottleneck, self.num_stacks, self.num_blocks, self.depth_res)
        vox2coord = coordRegressor(self.nParts)

        if self.is_cuda:
            self.pix2vox = torch.nn.DataParallel(pix2vox).cuda()
            self.vox2coord = torch.nn.DataParallel(vox2coord).cuda()
        else:
            self.pix2vox = pix2vox
            self.vox2coord = vox2coord

        print('    p2v params: %.2fM' % (sum(p.numel() for p in self.pix2vox.parameters()) / 1000000.0))
        print('    v2c params: %.2fM' % (sum(p.numel() for p in self.vox2coord.parameters()) / 1000000.0))


    def forward(self, x):
        vox_list = self.pix2vox(x)
        voxel = vox_list[-1].unsqueeze(1)
        output_ae = self.vox2coord(voxel)
        return vox_list, voxel, output_ae


    def landmarkDetection(self, image, target=None):

        input_var = torch.autograd.Variable(image, volatile=True)
        output_fl, input_ae_fl, output_ae_fl = self.forward(input_var)

        pred_cord_fl = 255*output_ae_fl[-1].view(-1, 3)

        return output_fl, pred_cord_fl


    def resume_from_checkpoint(self):
        if isfile(self.resume['p2v2c']):
            print("=> loading p2v2c checkpoint '{}'".format(self.resume['p2v2c']))
            if self.is_cuda:
                checkpoint_p2v2c = torch.load(self.resume['p2v2c'])
                self.pix2vox.load_state_dict(checkpoint_p2v2c['state_dict_p2v'])
                self.vox2coord.load_state_dict(checkpoint_p2v2c['state_dict_v2c'])
            else:
                # load onto the CPU
                checkpoint_p2v2c = torch.load(self.resume['p2v2c'], map_location=lambda storage, loc: storage)
                # remove module. from dict keys
                checkpoint_p2v = {k[7:]: v for k, v in checkpoint_p2v2c['state_dict_p2v'].items()}
                checkpoint_v2c = {k[7:]: v for k, v in checkpoint_p2v2c['state_dict_v2c'].items()}
                self.pix2vox.load_state_dict(checkpoint_p2v)
                self.vox2coord.load_state_dict(checkpoint_v2c)
            print("=> loaded checkpoint '{}'".format(self.resume['p2v2c']))
        else:
            # load pixel2voxel model
            if isfile(self.resume['p2v']):
                print("=> loading p2v checkpoint '{}'".format(self.resume['p2v']))
                checkpoint_p2v = torch.load(self.resume['p2v'])
                self.pix2vox.load_state_dict(checkpoint_p2v['state_dict_p2v'])
                print("=> loaded checkpoint '{}'".format(self.resume['p2v']))

            # load vox2coord model
            if isfile(self.resume['v2c']):
                print("=> loading checkpoint '{}'".format(self.resume['v2c']))
                checkpoint_v2c = torch.load(self.resume['v2c'])
                self.vox2coord.load_state_dict(checkpoint_v2c['state_dict_v2c'])
                print("=> loaded checkpoint '{}'".format(self.resume['v2c']))


    def save_to_checkpoint(self, predictions, checkpoint, snapshot=1):
        is_best = self.current_acc > self.best_acc
        if is_best:
            print('new record:{}'.format(self.current_acc))
            self.best_acc = self.current_acc
        save_checkpoint({
            'arch': 'p2v2c',
            'epoch': self.epoch,
            'state_dict_p2v': self.pix2vox.state_dict(),
            'state_dict_v2c': self.vox2coord.state_dict(),
        }, predictions, is_best, checkpoint=checkpoint, snapshot=snapshot)
