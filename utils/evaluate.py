import numpy as np
from collections import OrderedDict
import pickle
import gzip
from .utils import expand_frame_label, parse_label, easy_reduce

def levenstein(p, y, norm=False):
    m_row = len(p)    
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], np.float64)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)
    
    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score

def segs_to_labels_start_end_time(seg_list, bg_class):
    seg_list = [ s for s in seg_list if s.action not in bg_class ]
    labels = [ p.action for p in seg_list ]
    start  = [ p.start for p in seg_list ]
    end    = [ p.end+1 for p in seg_list ]
    return labels, start, end

def edit_score(pred_segs, gt_segs, norm=True, bg_class=["background"]):
    P, _, _ = segs_to_labels_start_end_time(pred_segs, bg_class)
    Y, _, _ = segs_to_labels_start_end_time(gt_segs, bg_class)
    return levenstein(P, Y, norm)

def f_score(pred_segs, gt_segs, overlap, bg_class=["background"]):
    p_label, p_start, p_end = segs_to_labels_start_end_time(pred_segs, bg_class)
    y_label, y_start, y_end = segs_to_labels_start_end_time(gt_segs, bg_class)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1

    fn = len(y_label) - sum(hits)

    return float(tp), float(fp), float(fn)


class Video():
    
    def __init__(self, vname='', **kwargs):
        self.vname = vname
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def __str__(self):
        return "< Video %s >" % self.vname

    def __repr__(self):
        return "< Video %s >" % self.vname

class Checkpoint():
    def __init__(self, iteration, bg_class=[], eval_edit=True):
        self.iteration = iteration
        self.videos = {}

        self.bg_class = bg_class
        self.eval_edit = eval_edit # turn off this can save computation time

    def add_videos(self, videos: list):
        for v in videos:
            self.videos[v.vname] = v

    @staticmethod
    def load(fname):
        with gzip.open(fname, 'rb') as fp:
            ckpt = pickle.load(fp)
        return ckpt
    
    def save(self, fname):
        self.fname = fname
        with gzip.open(fname, 'wb') as fp:
            pickle.dump(self, fp)

    def __str__(self):
        return "< Checkpoint[%d] %d videos >" % (self.iteration, len(self.videos))

    def __repr__(self):
        return str(self)

    def _random_video(self):
        vnames = list(self.videos.keys())
        vname = np.random.choice(vnames, 1).item()
        return vname, self.videos[vname]

    def average_losses(self):
        losses = [v.loss for v in self.videos.values()]
        self.loss = easy_reduce(losses, mode='mean')

    def _per_video_metrics(self, gt_label, pred_label):

        M = OrderedDict()

        if self.eval_edit:
            pred_segs = parse_label(pred_label)
            gt_segs = parse_label(gt_label)
            M['Edit'] = edit_score(pred_segs, gt_segs, bg_class=self.bg_class)

        return M

    def _joint_metrics(self, gt_list, pred_list):
        M = OrderedDict()

        # framewise accuracy
        gt_ = np.concatenate(gt_list)
        pred_ = np.concatenate(pred_list)

        correct = (gt_ == pred_)
        fg_loc = np.array([ True if g not in self.bg_class else False for g in gt_ ])
        M['AccB'] = correct.mean() * 100 # accuracy including background frames
        M['Acc'] = correct[fg_loc].mean() * 100 # accuracy without background frames

        # F1-Score
        overlap = [.1, .25, .5]
        tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

        for gt, pred in zip(gt_list, pred_list):
            gt_segs = parse_label(gt)
            pred_segs = parse_label(pred)
            for s in range(len(overlap)):
                tp1, fp1, fn1 = f_score(pred_segs, gt_segs, overlap[s], bg_class=self.bg_class)
                tp[s] += tp1
                fp[s] += fp1
                fn[s] += fn1

        for s in range(len(overlap)):
            precision = tp[s] / float(tp[s]+fp[s]+1e-5)
            recall = tp[s] / float(tp[s]+fn[s]+1e-5)
            f1 = 2.0 * (precision*recall) / (precision+recall+1e-5)
            f1 = np.nan_to_num(f1)*100
            M['F1@%0.2f' % overlap[s]] = f1

        return M

    def compute_metrics(self):

        gt_list, pred_list = [], [] 
        for vname, video in self.videos.items():
            video.pred_label = expand_frame_label(video.pred, len(video.gt_label)) # adjust for downsampling
            video.metrics = self._per_video_metrics(video.gt_label, video.pred_label)
            gt_list.append(video.gt_label)
            pred_list.append(video.pred_label)

        metrics = [ video.metrics for video in self.videos.values() ]
        self.metrics = easy_reduce(metrics, skip_nan=True)
        m = self._joint_metrics(gt_list, pred_list)
        self.metrics.update(m)

        return self.metrics
