import torch
import torch.nn as nn
import torch.nn.functional as F
from . import basic as basic
from ..utils import utils
from ..configs.utils import update_from
from . import loss
from .loss import MatchCriterion
from .basic import torch_class_label_to_segment_label, time_mask

_VIDS = None
_NIDS = None

class FACT(nn.Module):

    def __init__(self, cfg, in_dim, n_classes1=98, n_classes2=301):
        super().__init__()

        init_VIDS_NIDS()
        assert _VIDS and _NIDS

        self.cfg = cfg
        self.num_classes1 = n_classes1 # number of verb classes
        self.num_classes2 = n_classes2 # number of noun classes

        base_cfg = cfg.Bi

        self.frame_pe = basic.PositionalEncoding(base_cfg.hid_dim, max_len=10000, empty=(not cfg.FACT.fpos) )
        self.channel_masking_dropout = nn.Dropout2d(p=cfg.FACT.cmr)

        if not cfg.FACT.trans:
            self.action_query = nn.Parameter(torch.randn([cfg.FACT.ntoken, 1, base_cfg.a_dim]))
        else:
            self.action_pe = basic.PositionalEncoding(base_cfg.a_dim, max_len=1000)
            self.verb_embed = nn.Embedding(n_classes1, base_cfg.a_dim//2)
            self.noun_embed = nn.Embedding(n_classes2, base_cfg.a_dim//2)

        # block configuration
        block_list = []
        for i, t in enumerate(cfg.FACT.block):

            if t == 'I':
                block = InputBlockTDU(cfg, in_dim, n_classes1, n_classes2)
            elif t == 'U':
                update_from(cfg.BU, base_cfg, inplace=True)
                base_cfg = cfg.BU
                # prev_block = [ block ] # TODO
                block = UpdateBlockTDU(cfg, n_classes1, n_classes2)
                # block.prev_block = prev_block # TODO
            else:
                raise ValueError(t)

            block_list.append(block)

        self.block_list = nn.ModuleList(block_list)

        self.mcriterion = None

    def _forward_one_video(self, seq, transcript=None):
        # prepare frame feature
        frame_feature = seq
        frame_pe = self.frame_pe(seq)
        if self.cfg.FACT.cmr:
            frame_feature = frame_feature.permute([1, 2, 0]) # 1, H, T
            frame_feature = self.channel_masking_dropout(frame_feature)
            frame_feature = frame_feature.permute([2, 0, 1])

        if self.cfg.TM.use and self.training:
            frame_feature = time_mask(frame_feature, 
                        self.cfg.TM.t, self.cfg.TM.m, self.cfg.TM.p, 
                        replace_with_zero=True)

        # prepare action feature
        if not self.cfg.FACT.trans:
            action_pe = self.action_query # Q, 1, H
            action_feature = torch.zeros_like(action_pe)
        else:
            vtranscript = torch.LongTensor([_VIDS[a] for a in transcript ]).to(transcript.device) 
            ntranscript = torch.LongTensor([_NIDS[a] for a in transcript ]).to(transcript.device)
            action_pe = self.action_pe(transcript)
            vfeature = self.verb_embed(vtranscript).unsqueeze(1)
            nfeature = self.noun_embed(ntranscript).unsqueeze(1)
            action_feature = torch.cat([vfeature, nfeature], dim=-1)

            action_feature = action_feature + action_pe
            action_pe = torch.zeros_like(action_pe)

        # forward
        block_output = []
        for i, block in enumerate(self.block_list):
            output = block(frame_feature, action_feature, frame_pe, action_pe)
            block_output.append(output)
            frame_feature, action_feature = output
        return block_output

    def _loss_one_video(self, label):
        ########### Loss Section
        mcriterion: MatchCriterion = self.mcriterion
        mcriterion.set_label(label)

        block : Block = self.block_list[-1]
        cprob = torch.exp(block.action_logp)
        match = mcriterion.match(cprob, block.a2f_attn)

        ######## per block loss
        loss_list = []
        for block in self.block_list:
            loss = block.compute_loss(mcriterion, match)
            loss_list.append(loss)

        self.loss_list = loss_list
        final_loss = sum(loss_list) / len(loss_list)
        return final_loss

    def forward(self, seq_list, label_list, compute_loss=False):

        save_list = []
        final_loss = []

        for i, (seq, label) in enumerate(zip(seq_list, label_list)):
            seq = seq.unsqueeze(1)
            trans = torch_class_label_to_segment_label(label)[0]
            self._forward_one_video(seq, trans)

            pred = self.block_list[-1].eval(trans)
            save_data = {'pred': utils.to_numpy(pred)}
            save_list.append(save_data)

            if compute_loss:
                loss = self._loss_one_video(label)
                final_loss.append(loss)
                save_data['loss'] = { "loss": loss.item() }

        
        if compute_loss:
            final_loss = sum(final_loss) / len(final_loss)
            return final_loss, save_list
        else:
            return save_list

    def save_model(self, fname):
        torch.save(self.state_dict(), fname)

####################################################################
# helper function


def init_VIDS_NIDS():
    """
    for each action class in epic-kitchen, compute the ids of their correspondent words and nouns
    """
    global _VIDS, _NIDS
    if (_VIDS is None) or (_NIDS is None):
        from ..utils.dataset import load_action_mapping

        v2i, i2v = load_action_mapping('./data/epic-kitchens/processed/verb_mapping.txt')
        n2i, i2n = load_action_mapping('./data/epic-kitchens/processed/noun_mapping.txt')

        fname = './data/epic-kitchens/processed/mapping.txt'
        _VIDS = []
        _NIDS = []
        with open(fname) as fp:
            content = fp.read()
        lines = content.split('\n')[:-1]
        for l in lines:
            aid, aname = l.split(' ')
            v, n = aname.split(',')

            _VIDS.append(v2i[v])
            _NIDS.append(n2i[n])


####################################################################
# Blocks


class Block(nn.Module):

    def __init__(self):
        super().__init__()

    def __str__(self):
        lines = f"{type(self).__name__}(\n  f:{self.frame_branch},\n  a:{self.action_branch},\n  a2f:{self.a2f_layer if hasattr(self, 'a2f_layer') else None},\n  f2a:{self.f2a_layer if hasattr(self, 'f2a_layer') else None}\n)"
        return lines

    def __repr__(self):
        return str(self)

    def combine_verb_noun_to_action(self, clogit, action=False, apply_log=False):
        """
        combine the class logit of verbs and nouns to compute the probability of actions (verb + noun)
        clogit: T, B, C
        """
        global _VIDS, _NIDS
        nclass1 = self.nclass1 # number of verbs
        nclass2 = self.nclass2 # number of nouns

        if action:
            vlogit = clogit[..., :nclass1+1]
            nlogit = clogit[..., nclass1+1:]
            assert vlogit.shape[-1] == (max(_VIDS) + 2)
            assert nlogit.shape[-1] == (max(_NIDS) + 2)
        else:
            vlogit = clogit[..., :nclass1]
            nlogit = clogit[..., nclass1:]
            assert vlogit.shape[-1] == max(_VIDS) + 1
            assert nlogit.shape[-1] == max(_NIDS) + 1

        if not apply_log:
            v = torch.softmax(vlogit, dim=-1)
            n = torch.softmax(nlogit, dim=-1)
            a = v[..., _VIDS] * n[..., _NIDS]
            if action:
                null = (v[..., -1] * n[..., -1]).unsqueeze(-1)
                a = torch.cat([a, null], dim=-1)
        else:
            v = torch.log_softmax(vlogit, dim=-1)
            n = torch.log_softmax(nlogit, dim=-1)
            a = v[..., _VIDS] + n[..., _NIDS]
            if action:
                null = (v[..., -1] + n[..., -1]).unsqueeze(-1)
                a = torch.cat([a, null], dim=-1)

        return a


    def process_feature(self, feature, nclass1, nclass2):
        clogit = feature[:, :, -nclass1-nclass2:]
        feature = feature[:, :, :-nclass1-nclass2]
        cprob = basic.logit2prob(clogit, dim=-1, class_sep=nclass1)
        feature = torch.cat([feature, cprob], dim=-1)
        return feature, clogit

    def create_fbranch(self, cfg, in_dim=None, f_inmap=False):
        if in_dim is None:
            in_dim = cfg.f_dim

        if cfg.f == 'm': # use MSTCN
            frame_branch = basic.MSTCN(in_dim, cfg.f_dim, cfg.hid_dim, cfg.f_layers, 
                                dropout=cfg.dropout, ln=cfg.f_ln, ngroup=cfg.f_ngp, in_map=f_inmap)
        elif cfg.f == 'm2': # use MSTCN++
            frame_branch = basic.MSTCN2(in_dim, cfg.f_dim, cfg.hid_dim, cfg.f_layers, 
                                dropout=cfg.dropout, ln=cfg.f_ln, ngroup=cfg.f_ngp, in_map=f_inmap)

        return frame_branch

    def create_abranch(self, cfg):
        if cfg.a == 'sa':
            l = basic.SALayer(cfg.a_dim, cfg.a_nhead, dim_feedforward=cfg.a_ffdim, dropout=cfg.dropout, attn_dropout=cfg.dropout)
            action_branch = basic.SADecoder(cfg.a_dim, cfg.a_dim, cfg.hid_dim, l, cfg.a_layers, in_map=False)
        elif cfg.a == 'sca':
            layer = basic.SCALayer(cfg.a_dim, cfg.hid_dim, cfg.a_nhead, cfg.a_ffdim, dropout=cfg.dropout, attn_dropout=cfg.dropout)
            norm = torch.nn.LayerNorm(cfg.a_dim)
            action_branch = basic.SCADecoder(cfg.a_dim, cfg.a_dim, cfg.hid_dim, layer, cfg.a_layers, norm=norm, in_map=False)
        elif cfg.a in ['gru', 'gru_om']:
            assert self.cfg.FACT.trans
            out_map = (cfg.a == 'gru_om')
            action_branch = basic.ActionUpdate_GRU(cfg.a_dim, cfg.a_dim, cfg.hid_dim, cfg.a_layers, dropout=cfg.dropout, out_map=out_map)
        else:
            raise ValueError(cfg.a)

        return action_branch

    def create_cross_attention(self, cfg, outdim, kq_pos=True):

        layer = basic.X2Y_map(cfg.hid_dim, cfg.hid_dim, outdim, 
                head_dim=cfg.hid_dim, dropout=cfg.dropout, kq_pos=kq_pos)
        
        return layer

    def action_token_loss(self, criterion: loss.MatchCriterion, match, action_logp):
        logp = action_logp.squeeze(1)
        aind, sind = match
        A, C = logp.shape[0], logp.shape[-1]

        # action prediction loss
        clabel = torch.zeros(A, C).to(logp.device).long()
        clabel[:, -1] = 1 
        clabel[aind] = 0
        clabel[aind, criterion.transcript[sind]] = 1 #criterion.transcript[sind]

        qtk_loss = ((- logp * clabel) * criterion.cweight).sum(-1).mean()
        return qtk_loss

    def temporal_downsample(self, frame_feature):

        cprob = frame_feature[..., -self.nclass1-self.nclass2:]
        vprob, nprob = cprob[..., :self.nclass1], cprob[..., self.nclass1:]
        cprob = vprob[..., _VIDS] * nprob[..., _NIDS]

        # get action segments based on predictions
        maxp, pred = cprob[:, 0].max(dim=-1)
        pred = utils.to_numpy(pred)
        segs = utils.parse_label(pred)

        tdu = basic.TemporalDownsampleUpsample(segs)
        tdu.to(cprob.device)

        # downsample frames to segments
        seg_feature = tdu.feature_frame2seg(frame_feature)

        # refine segment features
        seg_feature, hidden = self.seg_update(seg_feature)
        seg_feature = torch.relu(seg_feature)
        seg_feature = self.seg_combine(seg_feature) # combine forward and backward features of GRU
        seg_feature, seg_clogit = self.process_feature(seg_feature, self.nclass1, self.nclass2)

        return tdu, seg_feature, seg_clogit

    def temporal_upsample(self, tdu, seg_feature, frame_feature):

        # upsample segments to frames
        s2f_fl = tdu.feature_seg2frame(seg_feature)
        
        # merge with original framewise features to keep low-level details
        frame_feature = self.sf_merge(torch.cat([s2f_fl, frame_feature], dim=-1))

        return frame_feature


    
    # @staticmethod
    def _eval(self, action_logp, a2f_attn, frame_logp, weight):
        fbranch_prob = torch.exp(frame_logp.squeeze(1))

        action_logp = action_logp.squeeze(1)
        a2f_attn = a2f_attn.squeeze(0)
        qtk_cpred = action_logp.argmax(1) 
        null_cid = action_logp.shape[-1] - 1
        action_loc = torch.where(qtk_cpred!=null_cid)[0]

        if len(action_loc) == 0:
            return fbranch_prob.argmax(1)

        qtk_prob = torch.exp(action_logp[:, :-1]) 
        qtk_prob = qtk_prob / qtk_prob.sum(-1, keepdims=True)
        action_pred = a2f_attn[:, action_loc].argmax(-1)
        action_pred = action_loc[action_pred]
        abranch_prob = qtk_prob[action_pred]

        prob = (1-weight) * abranch_prob + weight * fbranch_prob
        return prob.argmax(1)

    def _eval_w_transcript(self, transcript, a2f_attn):
        N = len(transcript)
        a2f_attn = a2f_attn[0, :, :N] # 1, f, a -> f, s'
        pred = a2f_attn.argmax(1) # f
        pred = transcript[pred]
        return pred

    def eval(self, transcript=None):
        if not self.cfg.FACT.trans:
            return self._eval(self.action_logp, self.a2f_attn, self.frame_logp, self.cfg.FACT.mwt)
        else:
            return self._eval_w_transcript(transcript, self.a2f_attn)


class InputBlockTDU(Block):
    def __init__(self, cfg, in_dim, nclass1, nclass2):
        super().__init__()
        self.cfg = cfg
        self.nclass1 = nclass1
        self.nclass2 = nclass2 

        cfg = cfg.Bi

        self.frame_branch = self.create_fbranch(cfg, in_dim, f_inmap=True)
        self.action_branch = self.create_abranch(cfg)
        
        self.seg_update = nn.GRU(cfg.hid_dim, cfg.hid_dim//2, 2, bidirectional=True)
        self.seg_combine = nn.Linear(cfg.hid_dim, cfg.hid_dim)

    def forward(self, frame_feature, action_feature, frame_pos, action_pos):
        """
        frame_feature: T, 1, H
        action_feature: N, 1, H
        """

        # frame branch
        frame_feature = self.frame_branch(frame_feature)
        frame_feature, frame_clogit = self.process_feature(frame_feature, self.nclass1, self.nclass2)

        # temporal downsample to improve training speed
        tdu, seg_feature, seg_clogit = self.temporal_downsample(frame_feature)

        # action branch
        center = torch.LongTensor([ int( (s.start+s.end)/2 ) for s in tdu.segs ]).to(seg_feature.device)
        seg_pos = frame_pos[center]
        action_feature = self.action_branch(action_feature, seg_feature, pos=seg_pos, query_pos=action_pos)
        action_feature, action_clogit = self.process_feature(action_feature, self.nclass1+1, self.nclass2+1)

        self.seg_logp = self.combine_verb_noun_to_action(seg_clogit, apply_log=True)
        self.frame_logp = self.combine_verb_noun_to_action(frame_clogit, apply_log=True)
        self.action_logp = self.combine_verb_noun_to_action(action_clogit, action=True, apply_log=True)
        self.tdu = tdu
        self.f2a_attn = None
        self.a2f_attn = None # self.a2f_layer.attn[0]
        return frame_feature, action_feature

    def compute_loss(self, criterion: loss.MatchCriterion, match=None):
        # frame loss
        frame_loss = criterion.frame_loss(self.frame_logp.squeeze(1), is_logit=False) / 2
        seg_loss = criterion.frame_loss_tdu(self.seg_logp.squeeze(1), self.tdu, is_logit=False) / 2

        # action token loss
        atk_loss = self.action_token_loss(criterion, match, self.action_logp) / 2
        # atk_loss_ref = criterion.action_token_loss(match, self.action_logp, is_logit=False) / 2

        # smooth loss
        logp = torch.transpose(self.frame_logp, 0, 1) # f, 1, c -> 1, f, c
        sl = loss.smooth_loss( logp, is_logit=False )

        return (frame_loss + seg_loss) / 2 + atk_loss + self.cfg.Loss.sw * sl

class UpdateBlockTDU(Block):
    """
    Update Block with Temporal Downsampling and Upsampling
    """

    def __init__(self, cfg, nclass1, nclass2):
        super().__init__()
        self.cfg = cfg
        self.nclass1 = nclass1
        self.nclass2 = nclass2

        cfg = cfg.BU

        # fbranch
        self.frame_branch = self.create_fbranch(cfg)

        # layers for temporal downsample and upsample
        self.seg_update = nn.GRU(cfg.hid_dim, cfg.hid_dim//2, cfg.s_layers, bidirectional=True)
        self.seg_combine = nn.Linear(cfg.hid_dim, cfg.hid_dim)

        # f2a: query is action
        self.f2a_layer = self.create_cross_attention(cfg, cfg.a_dim)

        # abranch
        self.action_branch = self.create_abranch(cfg)

        # a2f: query is frame
        self.a2f_layer = self.create_cross_attention(cfg, cfg.f_dim)

        self.sf_merge = nn.Linear((cfg.hid_dim+cfg.f_dim), cfg.f_dim) 
        self.sf_merge = nn.Sequential(nn.Linear((cfg.hid_dim+cfg.f_dim), cfg.f_dim), nn.ReLU())


    def forward(self, frame_feature, action_feature, frame_pos, action_pos):
        # downsample frame features to segment features
        tdu, seg_feature, seg_clogit = self.temporal_downsample(frame_feature) # seg_feature: S, 1, H

        # f->a
        seg_center = torch.LongTensor([ int( (s.start+s.end)/2 ) for s in tdu.segs ]).to(seg_feature.device)
        seg_pos = frame_pos[seg_center]
        action_feature = self.f2a_layer(seg_feature, action_feature, X_pos=seg_pos, Y_pos=action_pos)

        # a branch
        action_feature = self.action_branch(action_feature, action_pos)
        action_feature, action_clogit = self.process_feature(action_feature, self.nclass1+1, self.nclass2+1)

        # a->f
        seg_feature = self.a2f_layer(action_feature, seg_feature, X_pos=action_pos, Y_pos=seg_pos)

        # upsample segment features to frame features
        frame_feature = self.temporal_upsample(tdu, seg_feature, frame_feature)

        # f branch
        frame_feature = self.frame_branch(frame_feature)
        frame_feature, frame_clogit = self.process_feature(frame_feature, self.nclass1, self.nclass2)

        # save features for loss and evaluation       
        self.seg_logp = self.combine_verb_noun_to_action(seg_clogit, apply_log=True)
        self.frame_logp = self.combine_verb_noun_to_action(frame_clogit, apply_log=True)
        self.action_logp = self.combine_verb_noun_to_action(action_clogit, action=True, apply_log=True)
        self.tdu = tdu

        self.f2a_attn_logit = self.f2a_layer.attn_logit[0].unsqueeze(0)
        self.f2a_attn = tdu.attn_seg2frame(self.f2a_layer.attn[0].transpose(2, 1)).transpose(2, 1)
        self.a2f_attn_logit = self.a2f_layer.attn_logit[0].unsqueeze(0) # 1, a, s
        self.a2f_attn = tdu.attn_seg2frame(self.a2f_layer.attn[0])

        return frame_feature, action_feature

    def compute_loss(self, criterion: MatchCriterion, match=None):
        # frame loss
        frame_loss = criterion.frame_loss(self.frame_logp.squeeze(1), is_logit=False) / 2
        seg_loss = criterion.frame_loss_tdu(self.seg_logp.squeeze(1), self.tdu, is_logit=False) / 2

        atk_loss = self.action_token_loss(criterion, match, self.action_logp) / 2
        f2a_loss = criterion.cross_attn_loss_tdu(match, torch.transpose(self.f2a_attn_logit, 1, 2), self.tdu, dim=1)
        a2f_loss = criterion.cross_attn_loss_tdu(match, self.a2f_attn_logit, self.tdu, dim=2)

        # smooth loss
        sl = loss.smooth_loss( torch.transpose(self.frame_logp, 0, 1), is_logit=False )

        return (frame_loss + seg_loss) / 2 + atk_loss + f2a_loss + a2f_loss + self.cfg.Loss.sw * sl



