from __future__ import absolute_import

from .misc import *
from .transforms import transform, transform_preds

__all__ = ['accuracy', 'AverageMeter']

def get_preds(scores):
    ''' get predictions from score maps in torch Tensor
        return type: torch.LongTensor
    '''
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:,:,0] = (preds[:,:,0] - 1) % scores.size(3) + 1
    preds[:,:,1] = torch.floor((preds[:,:,1] - 1) / scores.size(2)) + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds

def calc_dists(preds, target, normalize):
    preds = preds.float()
    target = target.float()
    dists = torch.zeros(preds.size(1), preds.size(0))
    for n in range(preds.size(0)):
        for c in range(preds.size(1)):
            if target[n,c,0] > 1 and target[n, c, 1] > 1:
                dists[c, n] = torch.dist(preds[n,c,:], target[n,c,:])/normalize[n]
            else:
                dists[c, n] = -1
    return dists

def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    if dists.ne(-1).sum() > 0:
        return dists.le(thr).eq(dists.ne(-1)).sum()*1.0 / dists.ne(-1).sum()
    else:
        return -1

def accuracy(output, target, idxs, thr=0.5):
    ''' Calculate accuracy according to PCK, but uses ground truth heatmap rather than x,y locations
        First value to be returned is average accuracy across 'idxs', followed by individual accuracies
    '''
    preds   = get_preds(output)
    gts     = get_preds(target)
    norm    = torch.ones(preds.size(0))*output.size(3)/10
    dists   = calc_dists(preds, gts, norm)

    acc = torch.zeros(len(idxs)+1)
    avg_acc = 0
    cnt = 0

    for i in range(len(idxs)):
        acc[i+1] = dist_acc(dists[idxs[i]-1])
        if acc[i+1] >= 0: 
            avg_acc = avg_acc + acc[i+1]
            cnt += 1
            
    if cnt != 0:  
        acc[0] = avg_acc / cnt
    return acc

def final_preds(output, center, scale, res):
    coords = get_preds(output) # float type

    # pose-utils
    for n in range(coords.size(0)):
        for p in range(coords.size(1)):
            hm = output[n][p]
            px = int(math.floor(coords[n][p][0]))
            py = int(math.floor(coords[n][p][1]))
            if px > 1 and px < res[0] and py > 1 and py < res[1]:
                diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1]-hm[py - 2][px - 1]])
                coords[n][p] += diff.sign() * .25
    preds = coords.clone()

    # Transform back
    for i in range(coords.size(0)):
        preds[i] = transform_preds(coords[i], center[i], scale[i], res)

    if preds.dim() < 3:
        preds = preds.view(1, preds.size())

    return preds

def bboxNormMeanError(pred, target):
    def square_root(x):
        return np.sqrt(x[0]*x[1])

    pred_np = to_numpy(pred)
    target_np = to_numpy(target)

    bbox = [[np.min(target_np[i,:,0]), np.min(target_np[i,:,1]), np.max(target_np[i,:,0]), np.max(target_np[i,:,1])]
           for i in range(target_np.shape[0])]
    bbox = np.array(bbox)

    bbox_size = [square_root(bbox[i, 2:4] - bbox[i, 0:2]) for i in range(len(bbox))]
    num_samples = pred_np.shape[0]
    num_pts = pred_np.shape[1]

    NME = [np.sum(np.linalg.norm(pred_np[i]-target_np[i], axis=1))/(num_pts*bbox_size[i]) for i in range(num_samples)]
    return NME

def p2pNormMeanError(pred, target, norm_idx=None, z_zero_mean=False):

    pred_np = to_numpy(pred)
    target_np = to_numpy(target)

    num_samples = pred_np.shape[0]
    num_pts = pred_np.shape[1]

    if z_zero_mean:
        z_mean_gap = [np.mean(target_np[i,:,2]) - np.mean(pred_np[i,:,2])  for i in range(num_samples)]
        for i in range(num_samples):
            pred_np[i, :, 2] += z_mean_gap[i]

    if norm_idx is not None:
        normalization = [np.linalg.norm(target_np[i, norm_idx[0], :] - target_np[i, norm_idx[1], :]) for i in range(num_samples)]
    else:
        normalization = [np.linalg.norm(target_np[i,36,:]-target_np[i,45,:]) for i in range(num_samples)]

    NME = [np.sum(np.linalg.norm(pred_np[i]-target_np[i], axis=1))/(num_pts*normalization[i]) for i in range(num_samples)]
    return NME
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def readPts(ptspath):
    pts = []
    with open(ptspath, 'r') as file:
        lines = file.readlines()
        num_points = int(lines[1].split(' ')[1])
        for i in range(3,3+num_points):
            point = [float(num) for num in lines[i].split(' ')]
            pts.append(point)

    return np.array(pts)


def boundingbox(target_np):
    bbox = [np.min(target_np[:,0]), np.min(target_np[:,1]), np.max(target_np[:,0]), np.max(target_np[:,1])]
    bbox = np.array(bbox)

    bbox[2:4] = bbox[2:4] - bbox[0:2]

    center = bbox[0:2] + bbox[2:4] / 2.
    scale = bbox[2] / 200.

    return center, scale, bbox