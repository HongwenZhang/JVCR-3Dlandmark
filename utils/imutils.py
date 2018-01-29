from __future__ import absolute_import

import torch
import torch.nn as nn
import numpy as np
import scipy.misc

from .misc import *

def im_to_numpy(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0)) # H*W*C
    return img

def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1)) # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img

def load_image(img_path):
    # H x W x C => C x H x W
    return im_to_torch(scipy.misc.imread(img_path, mode='RGB'))

def resize(img, owidth, oheight):
    img = im_to_numpy(img)
    print('%f %f' % (img.min(), img.max()))
    img = scipy.misc.imresize(
            img,
            (oheight, owidth)
        )
    img = im_to_torch(img)
    print('%f %f' % (img.min(), img.max()))
    return img

# =============================================================================
# Helpful functions generating groundtruth labelmap 
# =============================================================================

def gaussian(shape=(7,7),sigma=1):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    return to_torch(h).float()

def draw_labelmap(img, pt, sigma, type='Gaussian'):
    # Draw a 2D gaussian 
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    img = to_numpy(img)

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return to_torch(img)

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return to_torch(img)

# Create compact volumetric representation
def draw_labelvolume(vol, pt, sigma, type='Gaussian'):
    # Draw a 2D gaussian 
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    vol = to_numpy(vol)
    img = img = np.zeros((vol.shape[1:]))

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return to_torch(img)

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    # extend to z-axis
    if vol.shape[0] == vol.shape[1]:
        z_gauss = g[x0]
    else:
        z_gauss = np.exp(- ((x - x0) ** 2) / (2 * sigma ** 2))

    z = np.uint8(pt[2])
    for i in range(len(z_gauss)):
        z_idx = z-x0+i
        if z_idx < 0 or z_idx >= vol.shape[0]:
            continue
        else:
            vol[z_idx] = z_gauss[i] * img

    return to_torch(vol)

# =============================================================================
# Helpful display functions
# =============================================================================

def gauss(x, a, b, c, d=0):
    return a * np.exp(-(x - b)**2 / (2 * c**2)) + d


def color_heatmap(x):
    x = to_numpy(x)
    color = np.zeros((x.shape[0],x.shape[1],3))
    color[:,:,0] = gauss(x, .5, .6, .2) + gauss(x, 1, .8, .3)
    color[:,:,1] = gauss(x, 1, .5, .3)
    color[:,:,2] = gauss(x, 1, .2, .3)
    color[color > 1] = 1
    color = (color * 255).astype(np.uint8)
    return color

def imshow(img):
    npimg = im_to_numpy(img*255).astype(np.uint8)
    plt.imshow(npimg)
    plt.axis('off')


def pts_show(pts, show_idx=False):

    for i in range(pts.size(0)):
        if pts.size(1) < 3 or pts[i, 2] > 0:
            plt.plot(pts[i, 0], pts[i, 1], 'yo')
            if show_idx:
                plt.text(pts[i, 0], pts[i, 1], str(i))
    plt.axis('off')

def show_voxel(pred_heatmap3d, ax=None):

    if ax is None:
        ax = plt.subplot(111, projection='3d')

    view_angle = (-160, 30)
    ht_map = pred_heatmap3d[0]
    density = ht_map.flatten()
    density = np.clip(density, 0, 1)
    density /= density.sum()
    selected_pt = np.random.choice(range(len(density)), 10000, p=density)
    pt3d = np.unravel_index(selected_pt, ht_map.shape)
    density_map = ht_map[pt3d]

    ax.set_aspect('equal')
    ax.scatter(pt3d[0], pt3d[2], pt3d[1], c=density_map, s=2, marker='.', linewidths=0)
    set_axes_equal(ax)
    # ax.set_xlabel('d', fontsize=10)
    # ax.set_ylabel('w', fontsize=10)
    # ax.set_zlabel('h', fontsize=10)
    ax.view_init(*view_angle)

    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.zaxis.set_ticks([])

    ax.set_xlabel('', fontsize=10)
    ax.set_ylabel('', fontsize=10)
    ax.set_zlabel('', fontsize=10)


def show_joints(img, pts, show_idx=False, pairs=None, ax=None):
    if ax is None:
        ax = plt.subplot(111)

    imshow(img)
    pts_np = pts.numpy()

    for i in range(pts.size(0)):
        if pts.size(1) < 3 or pts[i, 2] > 0:
            # plt.plot(pts[i, 0], pts[i, 1], 'bo')
            ax.scatter(pts[i, 0], pts[i, 1], s=5, c='c', edgecolors='b', linewidths=0.3)
            if show_idx:
                plt.text(pts[i, 0], pts[i, 1], str(i))
            if pairs is not None:
                for p in pairs:
                    ax.plot(pts_np[p, 0], pts_np[p, 1], c='b', linewidth=0.3)
    plt.axis('off')

def show_joints3D(predPts, pairs=None, ax=None):
    if ax is None:
        ax = plt.subplot(111, projection='3d')

    view_angle = (-160, 30)
    if predPts.shape[1] > 2:
        ax.scatter(predPts[:, 2], predPts[:, 0], predPts[:, 1], s=5, c='c', marker='o', edgecolors='b', linewidths=0.5)
        # ax_pred.scatter(predPts[0, 2], predPts[0, 0], predPts[0, 1], s=10, c='g', marker='*')
        if pairs is not None:
            for p in pairs:
                ax.plot(predPts[p, 2], predPts[p, 0], predPts[p, 1], c='b',  linewidth=0.5)
    else:
        ax.scatter([0] * predPts.shape[0], predPts[:, 0], predPts[:, 1], s=10, marker='*')
    ax.set_xlabel('z', fontsize=10)
    ax.set_ylabel('x', fontsize=10)
    ax.set_zlabel('y', fontsize=10)
    ax.view_init(*view_angle)

    ax.set_aspect('equal')
    set_axes_equal(ax)


def show_sample(inputs, target):
    num_sample = inputs.size(0)
    num_joints = target.size(1)
    height = target.size(2)
    width = target.size(3)

    for n in range(num_sample):
        inp = resize(inputs[n], width, height)
        out = inp
        for p in range(num_joints):
            tgt = inp*0.5 + color_heatmap(target[n,p,:,:])*0.5
            out = torch.cat((out, tgt), 2)
        
        imshow(out)
        plt.show()

def sample_with_heatmap(inp, out, num_rows=2, parts_to_show=None):
    inp = to_numpy(inp * 255)
    out = to_numpy(out)

    img = np.zeros((inp.shape[1], inp.shape[2], inp.shape[0]))
    for i in range(3):
        img[:, :, i] = inp[i, :, :]

    if parts_to_show is None:
        parts_to_show = np.arange(out.shape[0])

    # Generate a single image to display input/output pair
    num_cols = int(np.ceil(float(len(parts_to_show)) / num_rows))
    size = np.uint8(img.shape[0] / num_rows)

    full_img = np.zeros((img.shape[0], size * (num_cols + num_rows), 3), np.uint8)
    full_img[:img.shape[0], :img.shape[1]] = img

    inp_small = scipy.misc.imresize(img, [size, size])

    # Set up heatmap display for each part
    for i, part in enumerate(parts_to_show):
        part_idx = part
        out_resized = scipy.misc.imresize(out[part_idx], [size, size])
        out_resized = out_resized.astype(float)/255
        out_img = inp_small.copy() * .3
        color_hm = color_heatmap(out_resized)
        out_img += color_hm * .7

        col_offset = (i % num_cols + num_rows) * size
        row_offset = (i // num_cols) * size
        full_img[row_offset:row_offset + size, col_offset:col_offset + size] = out_img

    return full_img


def sample_with_stacked_heatmap(inp, out, num_rows=1, parts_to_show=None):
    inp = to_numpy(inp * 255)
    if isinstance(out, list):
        out = [to_numpy(out[i]) for i in range(len(out))]
    else:
        out = [out]
        out = [to_numpy(out[i]) for i in range(len(out))]

    img = np.zeros((inp.shape[1], inp.shape[2], inp.shape[0]))
    for i in range(3):
        img[:, :, i] = inp[i, :, :]

    if parts_to_show is None:
        # parts_to_show = np.arange(out.shape[0])
        parts_to_show = np.arange(len(out))

    # Generate a single image to display input/output pair
    # num_cols = int(np.ceil(float(len(parts_to_show)) / num_rows))
    # size = np.uint8(img.shape[0] / num_rows)
    num_cols = len(out)
    num_rows = 1
    size = np.uint16(img.shape[0])

    # full_img = np.zeros((img.shape[0], size * (num_cols + num_rows), 3), np.uint8)
    # full_img[:img.shape[0], :img.shape[1]] = img
    full_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    full_img[:img.shape[0], :img.shape[1]] = img

    inp_small = scipy.misc.imresize(img, [size, size])

    for i in range(len(out)):
        stacked_out = np.max(out[i], axis=0)

        # Set up heatmap display for each part
        # for i, part in enumerate(parts_to_show):
        #     part_idx = part
        out_resized = scipy.misc.imresize(stacked_out, [size, size])
        out_resized = out_resized.astype(float) / 255
        out_img = inp_small.copy() * .3
        color_hm = color_heatmap(out_resized)
        out_img += color_hm * .7
        out_img = np.uint8(out_img)

        profile = np.max(out[i], axis=2)
        profile = np.swapaxes(profile, 0, 1)
        profile_resized = scipy.misc.imresize(profile, float(size/profile.shape[0]))
        profile_resized = profile_resized.astype(float) / 255
        out_pf = color_heatmap(profile_resized)
        out_pf = np.uint8(out_pf)

        full_img = np.concatenate((full_img, out_img, out_pf), axis=1)


    # col_offset = size
    # row_offset = 0
    # full_img[row_offset:row_offset + size, col_offset:col_offset + size] = out_img

    return full_img


def batch_with_heatmap(inputs, outputs, mean=torch.Tensor([0.5, 0.5, 0.5]), num_rows=2, parts_to_show=None):
    batch_img = []
    for n in range(min(inputs.size(0), 4)):
        inp = inputs[n] + mean.view(3, 1, 1).expand_as(inputs[n])
        batch_img.append(
            sample_with_heatmap(inp.clamp(0, 1), outputs[n], num_rows=num_rows, parts_to_show=parts_to_show)
        )
    return np.concatenate(batch_img)


def batch_with_stacked_heatmap(inputs, outputs, mean=torch.Tensor([0, 0, 0]), num_rows=1, parts_to_show=None):
    batch_img = []
    for n in range(min(inputs.size(0), 4)):
        inp = inputs[n] + mean.view(3, 1, 1).expand_as(inputs[n])
        batch_img.append(
            sample_with_stacked_heatmap(inp.clamp(0, 1), outputs[n], num_rows=num_rows, parts_to_show=parts_to_show)
        )
    return np.concatenate(batch_img)