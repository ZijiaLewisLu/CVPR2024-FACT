import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from . import basic as basic
from ..utils import utils
import numpy as np

def smooth_loss(logit, is_logit=True):
    """
    logit: B, T, C
    """
    if is_logit:
        logsoft = F.log_softmax(logit, dim=2)
    else:
        logsoft = logit
    loss = torch.clamp((logsoft[:, 1:] - logsoft[:, :-1])**2, min=0, max=16)
    loss = loss.mean()
    return loss

def torch_class_label_to_segment_label(label):
    segment_label = torch.zeros_like(label)
    current = label[0]
    transcript = [label[0]]
    aid = 0
    for i, l in enumerate(label):
        if l == current:
            pass
        else:
            current = l
            aid += 1
            transcript.append(l)
        segment_label[i] = aid

    transcript = torch.LongTensor(transcript).to(label.device)
    
    return transcript, segment_label

def logit2prob(clogit, dim=-1, class_sep=None):
    if class_sep is None or class_sep<=0:
        cprob = torch.softmax(clogit, dim=dim)
    else:
        assert dim==-1, dim
        cprob1 = torch.softmax(clogit[..., :class_sep], dim=dim)
        cprob2 = torch.softmax(clogit[..., class_sep:], dim=dim)
        cprob = torch.cat([cprob1, cprob2], dim=dim)
    
    return cprob

class MatchCriterion():

    def __init__(self, cfg, nclasses, bg_ids=[], class_weight=None):
        self.cfg = cfg
        self.nclasses = nclasses
        self.bg_ids = bg_ids
        self._class_weight=class_weight


    def set_label(self, label):
        self.class_label = label
        self.transcript, self.seg_label = torch_class_label_to_segment_label(label)
        self.onehot_class_label = self._label_to_onehot(self.class_label, self.nclasses)
        self.onehot_seg_label = self._label_to_onehot(self.seg_label, len(self.transcript))

        # create class weight
        cweight = torch.ones(self.nclasses+1).to(label.device)
        cweight[-1] = self.cfg.Loss.nullw
        if self._class_weight is not None:
            for i in range(self.nclasses):
                cweight[i] = self._class_weight[i]
        else:
            for i in self.bg_ids:
                cweight[i] = self.cfg.Loss.bgw

        # create weight for each action segment based on class weight
        sweight = torch.ones_like(self.transcript, dtype=torch.float32)
        if self._class_weight is not None:
            for i, t in enumerate(self.transcript.tolist()):
                sweight[i] = self._class_weight[t]
        else:
            for i in self.bg_ids:
                sweight[self.transcript==i] = self.cfg.Loss.bgw

        self.cweight=cweight
        self.sweight=sweight

    def _label_to_onehot(self, label, nclass):
        onehot_label = torch.zeros(len(label), nclass).to(label.device)
        onehot_label[torch.arange(len(label)), label] = 1
        return onehot_label

    @classmethod
    def a2f_soft_iou(self, a2f_attn, onehot_seg_label):
        """
        a2f_attn: 1, f, a, sum over a == 1
        onehot_seg_label: f, s
        """
        a2f_attn = a2f_attn[0].unsqueeze(-1) # 1, f, a -> f, a, 1
        onehot_seg_label = onehot_seg_label.unsqueeze(1) # f, s -> f, 1, s
        a2f_attn_np = utils.to_numpy(a2f_attn)
        onehot_seg_label_np = utils.to_numpy(onehot_seg_label)
        overlap = np.einsum('tax,txs->as', a2f_attn_np, onehot_seg_label_np)
        union = np.minimum(a2f_attn_np + onehot_seg_label_np, 1.0).sum(0) # a,s 
        iou = np.nan_to_num(overlap / union, nan=0.0)

        del a2f_attn_np, onehot_seg_label_np
        return iou

    def match(self, clogit, a2f_attn):
        """
        clogit: a, 1, c  
        f2a_attn: 1, a, f
        a2f_attn: 1, f, a
        """
        assert clogit.shape[1] == 1 # batch_size == 1

        match_cfg = self.cfg.Loss
        transcript = self.transcript
        onehot_seg_label = self.onehot_seg_label

        # sequential matching between tokens and groundtruth segments
        if match_cfg.match == 'seq':
            A = clogit.shape[0]
            S = onehot_seg_label.shape[-1]
            assert A >= S, (A, S)
            action_ind = seg_ind = torch.as_tensor(list(range(S)), dtype=torch.int64)
            return action_ind, seg_ind

        # compute matching cost 
        cost = 0
        with torch.no_grad():
            if match_cfg.pc > 0:
                prob = clogit.squeeze(1)
                prob = torch.index_select(prob, 1, transcript) # a, s
                prob = utils.to_numpy(prob)
                cost -= match_cfg.pc * prob
            
            if match_cfg.a2fc > 0:
                a2f_iou = self.a2f_soft_iou(a2f_attn, onehot_seg_label)
                a2f_iou = utils.to_numpy(a2f_iou)
                cost -= match_cfg.a2fc * a2f_iou

        cost = utils.to_numpy(cost) # a, s

        # find optimal matching
        if match_cfg.match == 'o2o': # one-to-one matching
            action_ind, seg_ind = linear_sum_assignment(cost)
        elif match_cfg.match == 'o2m': # one-to-many matching
            action_ind, seg_ind = self._one_to_many_match(cost)

        action_ind = torch.as_tensor(action_ind, dtype=torch.int64) # id of action query token
        seg_ind    = torch.as_tensor(seg_ind, dtype=torch.int64) # groundtruth action label id

        return action_ind, seg_ind

    def _one_to_many_match(self, cost):
        transcript_np = utils.to_numpy(self.transcript)
        actions = np.unique(transcript_np)
        token2action_cost = []
        for a in actions:
            where = (transcript_np == a)
            score = cost[:, where]
            score = score.sum(1)
            token2action_cost.append(score)
        token2action_cost = np.stack(token2action_cost, axis=1)

        _aid, _cid = linear_sum_assignment(token2action_cost)
        
        unassign_aid = [ a for a in range(cost.shape[0]) if a not in _aid ] 
        unassign_cid = token2action_cost[unassign_aid].argmin(1)

        all_aid = np.array(_aid.tolist() + unassign_aid)
        all_cid = [ actions[i] for i in _cid.tolist() + unassign_cid.tolist() ]
        all_cid = np.array(all_cid)

        atoken_cid = np.zeros(cost.shape[0])
        atoken_cid[all_aid] = all_cid

        match = {}
        for a in actions:
            seg_where = np.where(transcript_np == a)[0]
            token_where = np.where(atoken_cid == a)[0]
            subset = cost[token_where][:, seg_where]
            assign = subset.argmin(0)

            for s, a in zip(seg_where, assign):
                match[s] = token_where[a]
                
        aid_new, sid_new = [], []
        for k, v in match.items():
            aid_new.append(v)
            sid_new.append(k)
        
        return aid_new, sid_new

    def action_token_loss(self, match, action_clogit, is_logit=True):
        aind, sind = match
        A, C = action_clogit.shape[0], action_clogit.shape[-1]

        # action prediction loss
        clabel = torch.zeros(A).to(action_clogit.device).long() + C - 1 # shape: a; default = empty_class
        clabel[aind] = self.transcript[sind]
        if is_logit:
            loss = F.cross_entropy(action_clogit.squeeze(1), clabel, weight=self.cweight)
        else:
            loss = F.nll_loss(action_clogit.squeeze(1), clabel, weight=self.cweight)

        return loss

    def cross_attn_loss(self, match, attn, dim=None):
        assert dim >= 1
        onehot_seg_label = self.onehot_seg_label
        aind, sind = match

        frame_tgt = onehot_seg_label[:, sind] # f, s
        attn = attn[0, :, aind] # f, s
        attn_logp = torch.log_softmax(attn, dim=dim-1)
        loss2 = - attn_logp * frame_tgt 
        if self.sweight is not None:
            loss2 = loss2 * self.sweight
        loss2 = loss2.sum(1).sum() / self.onehot_seg_label.sum()

        return loss2

    def cross_attn_loss_tdu(self, match, attn, tdu: basic.TemporalDownsampleUpsample, dim=None):
        assert dim >= 1
        onehot_seg_label = self.onehot_seg_label
        aind, sind = match

        # f, c -> s, c
        zoomed_label = torch.zeros([tdu.num_seg, onehot_seg_label.shape[1]], dtype=onehot_seg_label.dtype).to(onehot_seg_label.device) 
        zoomed_label.index_add_(0, tdu.seg_label, onehot_seg_label)
        zoomed_label = zoomed_label / tdu.seg_lens[:, None]

        frame_tgt = zoomed_label[:, sind] # s, n
        attn = attn[0, :, aind] # s, n
        attn_logp = torch.log_softmax(attn, dim=dim-1)

        loss2 = - attn_logp * frame_tgt 
        if self.sweight is not None:
            loss2 = loss2 * self.sweight

        loss2 = loss2.sum(1).sum() / zoomed_label.sum()

        return loss2

    def frame_loss(self, frame_clogit, is_logit=True):
        if is_logit:
            logp = torch.log_softmax(frame_clogit, dim=-1)
        else:
            logp = frame_clogit

        cweight = self.cweight[:frame_clogit.shape[-1]] # remove the weight for null class
        frame_loss = - logp * self.onehot_class_label
        frame_loss = frame_loss * cweight

        frame_loss = frame_loss.sum(-1).sum() / self.onehot_class_label.sum()

        return frame_loss

    def frame_loss_tdu(self, seg_clogit, tdu, is_logit=True):
        if is_logit:
            logp = torch.log_softmax(seg_clogit.squeeze(1), dim=-1)
        else:
            logp = seg_clogit.squeeze(1)


        ohl = self.onehot_class_label
        zoomed_label = torch.zeros([tdu.num_seg, ohl.shape[1]], dtype=ohl.dtype).to(ohl.device) 
        zoomed_label.index_add_(0, tdu.seg_label, ohl)
        zoomed_label = zoomed_label / tdu.seg_lens[:, None]
        seg_loss = ( - logp * zoomed_label )
        _cweight = self.cweight[:logp.shape[-1]] # remove the weight for null class
        seg_loss = (seg_loss * _cweight)

        seg_loss = seg_loss.sum(-1).sum() / zoomed_label.sum()

        return seg_loss
    