from __future__ import print_function, absolute_import

import os
import numpy as np
import json
import random
import math

import torch
import torch.utils.data as data

from utils.osutils import *
from utils.imutils import *
from utils.transforms import *


class fa68pt3D(data.Dataset):
    def __init__(self, jsonfile, img_folder, dataset_name, inp_res=256, out_res=64, depth_res=None, train=True, jitter=False,
                 sigma=1, c2f=True, nStack=1, scale_factor=0.08, rot_factor=10, label_type='Gaussian'):
        self.img_folder = img_folder  # root image folders
        self.is_train = train  # training set or test set
        self.jitter = jitter  # jitter the dada
        self.inp_res = inp_res
        self.out_res = out_res
        self.depth_res = depth_res
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.label_type = label_type
        self.c2f = c2f
        self.nStack = nStack

        self.dataset_name = dataset_name
        # self.valid_ldmk_idx = range(68)
        self.skeletons = [[i, i + 1] for i in range(16)] + \
                         [[i, i + 1] for i in range(17, 21)] + \
                         [[i, i + 1] for i in range(22, 26)] + \
                         [[i, i + 1] for i in range(36, 41)] + [[41, 36]] + \
                         [[i, i + 1] for i in range(42, 47)] + [[47, 42]] + \
                         [[i, i + 1] for i in range(27, 30)] + \
                         [[i, i + 1] for i in range(31, 35)] + \
                         [[i, i + 1] for i in range(48, 59)] + [[59, 48]] + \
                         [[i, i + 1] for i in range(60, 67)] + [[67, 60]]

        self.n_joints = 71   # 68 points plus eye and mouth center

        # turn off c2f if depth_res is not provided
        if depth_res is None:
            self.depth_res = [self.out_res]
            self.c2f = False
            # self.depth_res = 60
        if len(self.depth_res) == 1:
                self.c2f = False

        if self.c2f:
            assert len(self.depth_res) == self.nStack

        # create train/val split
        with open(jsonfile) as anno_file:
            self.anno = json.load(anno_file)

        self.train, self.valid = [], []
        for idx, val in enumerate(self.anno):
            if val['isValidation'] == True:
                self.valid.append(idx)
            else:
                self.train.append(idx)

        # debug
        # self.train = self.train[:50]
        # self.valid = self.valid[:100]


    def __getitem__(self, index):
        sf = self.scale_factor
        rf = self.rot_factor
        if self.is_train:
            a = self.anno[self.train[index]]
        else:
            a = self.anno[self.valid[index]]

        img_path = os.path.join(self.img_folder, a['img_paths'])
        pts = torch.Tensor(a['landmarks'])
        pts[:, 2] = -pts[:, 2]              # flip z value
        pts[:, 2] -= torch.mean(pts[:, 2])  # zero z-axis mean
        # face_center3D = (pts[2] + pts[15] + pts[33]) / 3
        # pts[:, 2] -= face_center3D[2]
        # pts[:, 0:2] -= 1  # Convert pts to zero based

        # c = torch.Tensor(a['objpos']) - 1
        c = torch.Tensor(a['objpos'])
        s = a['scale_provided']

        # Adjust center/scale slightly to avoid cropping limbs
        if c[0] != -1:
            # c[1] = c[1] + 15 * s
            s = s * 1.25

        # For single-person pose estimation with a centered/scaled figure
        img = load_image(img_path)  # CxHxW

        r = 0
        if self.jitter:
            s = s * torch.randn(1).mul_(sf).add_(1).clamp(1 - sf, 1 + sf)[0]
            r = torch.randn(1).mul_(rf).clamp(-2 * rf, 2 * rf)[0] if random.random() <= 0.6 else 0

            # Flip
            if random.random() <= 0.5:
                img = torch.from_numpy(fliplr(img.numpy())).float()
                pts = shufflelr(pts, width=img.size(2), dataset=self.dataset_name)
                c[0] = img.size(2) - c[0]

            # Color
            img[0, :, :].add_(random.uniform(-0.3, 0.3)).clamp_(0, 1)
            img[1, :, :].add_(random.uniform(-0.3, 0.3)).clamp_(0, 1)
            img[2, :, :].add_(random.uniform(-0.3, 0.3)).clamp_(0, 1)

        # augment landmarks based aflw protocol
        mouth_center = (pts[62, :] + pts[66, :]) / 2
        eye_left = (pts[37, :] + pts[38, :] + pts[40, :] + pts[41, :]) / 4
        eye_right = (pts[43, :] + pts[44, :] + pts[46, :] + pts[47, :]) / 4

        pts = torch.cat(
            (pts, eye_left.view(1, -1), eye_right.view(1, -1), mouth_center.view(1, -1)), dim=0)

        # show_joints(img, pts[:, :2], show_idx=True)
        # nparts = pts.size(0)

        # Prepare image and groundtruth map
        inp = crop(img, c, s, [self.inp_res, self.inp_res], rot=r)
        tpts_inp = pts.clone()
        for i in range(tpts_inp.size(0)):
            tpts_inp[i, 0:3] = to_torch(transform3d(tpts_inp[i, 0:3] + 1, c, s, [self.inp_res, self.inp_res],
                                                    self.inp_res, rot=r))

        # Generate ground truth
        target = []
        vox_idx = range(self.nStack) if self.c2f else [-1]
        # target = torch.FloatTensor()

        # compact volume
        for i in range(self.nStack):
            target_i = torch.zeros(self.depth_res[i], self.out_res, self.out_res)
            tpts = pts.clone()
            for j in range(tpts.size(0)):
                # if tpts[j, 2] > 0: # This is evil!!
                if tpts[j, 0] > 0:
                    target_j = torch.zeros(self.depth_res[i], self.out_res, self.out_res)
                    tpts[j, 0:3] = to_torch(transform3d(tpts[j, 0:3] + 1, c, s, [self.out_res, self.out_res],
                                                        self.depth_res[i], rot=r))
                    target_j = draw_labelvolume(target_j, tpts[j] - 1, self.sigma, type=self.label_type)
                    target_i = torch.max(target_i, target_j.float())

            target.append(target_i)
            # target = torch.cat((target, target_i))

        # Meta info
        meta = {'dataset': self.dataset_name,
                # 'valid_idx': self.valid_ldmk_idx,
                'index': index, 'center': c, 'scale': s,
                'pts': pts, 'tpts': tpts, 'tpts_inp': tpts_inp,
                'skeletons': self.skeletons}

        return inp, target, meta

    def __len__(self):
        if self.is_train:
            return len(self.train)
        else:
            return len(self.valid)