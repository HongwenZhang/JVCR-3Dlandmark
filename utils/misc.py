from __future__ import absolute_import

import os
import shutil
import torch 
import math
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib import cm

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def save_checkpoint(state, preds, is_best, checkpoint='checkpoint', filename='checkpoint_{}.pth.tar', snapshot=None):
    preds = to_numpy(preds)
    filepath = os.path.join(checkpoint, filename.format(state['arch']))
    torch.save(state, filepath)
    scipy.io.savemat(os.path.join(checkpoint, 'preds_{}.mat'.format(state['arch'])), mdict={'preds' : preds})

    if snapshot and state['epoch'] % snapshot == 0:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'checkpoint_{}_{}.pth.tar'.format(state['arch'],
                                                                                             state['epoch'])))

    if is_best:
        print('save best model: model_best_{}_{}.pth.tar'.format(state['arch'], state['epoch']))
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best_{}_{}.pth.tar'.format(state['arch'],
                                                                                             state['epoch'])))
        scipy.io.savemat(os.path.join(checkpoint, 'preds_best_{}_{}.mat'.format(state['arch'], state['epoch'])),
                         mdict={'preds' : preds})


def save_pred(preds, checkpoint='checkpoint', filename='preds_valid.mat'):
    preds = to_numpy(preds)
    filepath = os.path.join(checkpoint, filename)
    scipy.io.savemat(filepath, mdict={'preds' : preds})


def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


# https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])