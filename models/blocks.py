import torch
import torch.nn as nn
import torch.nn.functional as F
from . import basic as basic
from ..utils import utils
from ..configs.utils import update_from
from . import loss
from .loss import MatchCriterion
from .basic import torch_class_label_to_segment_label, time_mask

class FACT(nn.Module):

    def __init__(self, cfg, in_dim, n_classes):
        super().__init__()
        self.cfg = cfg
        self.num_classes = n_classes

        base_cfg = cfg.Bi
        self.frame_pe = basic.PositionalEncoding(base_cfg.hid_dim, max_len=10000, empty=(not cfg.FACT.fpos) )
        self.channel_masking_dropout = nn.Dropout2d(p=cfg.FACT.cmr)

        if not cfg.FACT.trans : # when video transcript is not available at training and inference
            self.action_query = nn.Parameter(torch.randn([cfg.FACT.ntoken, 1, base_cfg.a_dim]))
        else: # when video transcript is available
            self.action_pe = basic.PositionalEncoding(base_cfg.a_dim, max_len=1000)
            self.action_embed = nn.Embedding(n_classes, base_cfg.a_dim)

        # block configuration
        block_list = []
        for i, t in enumerate(cfg.FACT.block):
            if t == 'i':
                block = InputBlock(cfg, in_dim, n_classes)
            elif t == 'u':
                update_from(cfg.Bu, base_cfg, inplace=True)
                base_cfg = cfg.Bu
                block = UpdateBlock(cfg, n_classes)
            elif t == 'U':
                update_from(cfg.BU, base_cfg, inplace=True)
                base_cfg = cfg.BU
                block = UpdateBlockTDU(cfg, n_classes)

            block_list.append(block)

        self.block_list = nn.ModuleList(block_list)

        self.mcriterion = None

    def _forward_one_video(self, seq, transcript=None):
        # prepare frame feature
        frame_feature = seq
        frame_pe = self.frame_pe(seq)
        if self.cfg.FACT.cmr:
            frame_feature = frame_feature.permute([1, 2, 0])
            frame_feature = self.channel_masking_dropout(frame_feature)
            frame_feature = frame_feature.permute([2, 0, 1])

        if self.cfg.TM.use and self.training:
            frame_feature = time_mask(frame_feature, 
                        self.cfg.TM.t, self.cfg.TM.m, self.cfg.TM.p, 
                        replace_with_zero=True)

        # prepare action feature
        if not self.cfg.FACT.trans:
            action_pe = self.action_query # M, B(=1), H
            action_feature = torch.zeros_like(action_pe)
        else:
            action_pe = self.action_pe(transcript)
            action_feature = self.action_embed(transcript).unsqueeze(1)

            action_feature = action_feature + action_pe
            action_pe = torch.zeros_like(action_pe)

        # forward
        # frame_feature: T, B(=1), H
        # action_feature: M, B(=1), H
        block_output = []
        for i, block in enumerate(self.block_list):
            frame_feature, action_feature = block(frame_feature, action_feature, frame_pe, action_pe)
            block_output.append([frame_feature, action_feature])
        return block_output

    def _loss_one_video(self, label):
        mcriterion: MatchCriterion = self.mcriterion
        mcriterion.set_label(label)

        block : Block = self.block_list[-1]
        cprob = basic.logit2prob(block.action_clogit, dim=-1)
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
                save_data['loss'] = { 'loss': loss.item() }


        if compute_loss:
            final_loss = sum(final_loss) / len(final_loss)
            return final_loss, save_list
        else:
            return save_list

    def save_model(self, fname):
        torch.save(self.state_dict(), fname)

####################################################################
# Blocks

class Block(nn.Module):
    """
    Base Block class for common functions
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        lines = f"{type(self).__name__}(\n  f:{self.frame_branch},\n  a:{self.action_branch},\n  a2f:{self.a2f_layer if hasattr(self, 'a2f_layer') else None},\n  f2a:{self.f2a_layer if hasattr(self, 'f2a_layer') else None}\n)"
        return lines

    def __repr__(self):
        return str(self)

    def process_feature(self, feature, nclass):
        # use the last several dimension as logit of action classes
        clogit = feature[:, :, -nclass:] # class logit
        feature = feature[:, :, :-nclass] # feature without clogit
        cprob = basic.logit2prob(clogit, dim=-1)  # apply softmax
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
        if cfg.a == 'sa': # self-attention layers, for update blocks
            l = basic.SALayer(cfg.a_dim, cfg.a_nhead, dim_feedforward=cfg.a_ffdim, dropout=cfg.dropout, attn_dropout=cfg.dropout)
            action_branch = basic.SADecoder(cfg.a_dim, cfg.a_dim, cfg.hid_dim, l, cfg.a_layers, in_map=False)
        elif cfg.a == 'sca': # self+cross-attention layers, for input blocks when video transcripts are not available
            layer = basic.SCALayer(cfg.a_dim, cfg.hid_dim, cfg.a_nhead, cfg.a_ffdim, dropout=cfg.dropout, attn_dropout=cfg.dropout)
            norm = torch.nn.LayerNorm(cfg.a_dim)
            action_branch = basic.SCADecoder(cfg.a_dim, cfg.a_dim, cfg.hid_dim, layer, cfg.a_layers, norm=norm, in_map=False)
        elif cfg.a in ['gru', 'gru_om']: # GRU, for input blocks when video transcripts are available
            assert self.cfg.FACT.trans
            out_map = (cfg.a == 'gru_om')
            action_branch = basic.ActionUpdate_GRU(cfg.a_dim, cfg.a_dim, cfg.hid_dim, cfg.a_layers, dropout=cfg.dropout, out_map=out_map)
        else:
            raise ValueError(cfg.a)

        return action_branch

    def create_cross_attention(self, cfg, outdim, kq_pos=True):
        # one layer of cross-attention for cross-branch communication
        layer = basic.X2Y_map(cfg.hid_dim, cfg.hid_dim, outdim, 
            head_dim=cfg.hid_dim,
            dropout=cfg.dropout, kq_pos=kq_pos)
        
        return layer

    @staticmethod
    def _eval(action_clogit, a2f_attn, frame_clogit, weight):
        fbranch_prob = torch.softmax(frame_clogit.squeeze(1), dim=-1)

        action_clogit = action_clogit.squeeze(1)
        a2f_attn = a2f_attn.squeeze(0)
        qtk_cpred = action_clogit.argmax(1) 
        null_cid = action_clogit.shape[-1] - 1
        action_loc = torch.where(qtk_cpred!=null_cid)[0]

        if len(action_loc) == 0:
            return fbranch_prob.argmax(1)

        qtk_prob = torch.softmax(action_clogit[:, :-1], dim=1) # remove logit of null classes
        action_pred = a2f_attn[:, action_loc].argmax(-1)
        action_pred = action_loc[action_pred]
        abranch_prob = qtk_prob[action_pred]

        prob = (1-weight) * abranch_prob + weight * fbranch_prob
        return prob.argmax(1)

    @staticmethod
    def _eval_w_transcript(transcript, a2f_attn):
        N = len(transcript)
        a2f_attn = a2f_attn[0, :, :N] # 1, f, a -> f, s'
        pred = a2f_attn.argmax(1) # f
        pred = transcript[pred]
        return pred

    def eval(self, transcript=None):
        if not self.cfg.FACT.trans:
            return self._eval(self.action_clogit, self.a2f_attn, self.frame_clogit, self.cfg.FACT.mwt)
        else:
            return self._eval_w_transcript(transcript, self.a2f_attn)


class InputBlock(Block):
    def __init__(self, cfg, in_dim, nclass):
        super().__init__()
        self.cfg = cfg
        self.nclass = nclass

        cfg = cfg.Bi

        self.frame_branch = self.create_fbranch(cfg, in_dim, f_inmap=True)
        self.action_branch = self.create_abranch(cfg)

    def forward(self, frame_feature, action_feature, frame_pos, action_pos, action_clogit=None):
        # frame branch
        frame_feature = self.frame_branch(frame_feature)
        frame_feature, frame_clogit = self.process_feature(frame_feature, self.nclass)

        # action branch
        action_feature = self.action_branch(action_feature, frame_feature, pos=frame_pos, query_pos=action_pos)
        action_feature, action_clogit = self.process_feature(action_feature, self.nclass+1)
        
        # save features for loss and evaluation
        self.frame_clogit = frame_clogit 
        self.action_clogit = action_clogit

        return frame_feature, action_feature

    def compute_loss(self, criterion: loss.MatchCriterion, match=None):
        frame_loss = criterion.frame_loss(self.frame_clogit.squeeze(1))
        atk_loss = criterion.action_token_loss(match, self.action_clogit)

        frame_clogit = torch.transpose(self.frame_clogit, 0, 1) 
        smooth_loss = loss.smooth_loss(frame_clogit)

        return frame_loss + atk_loss + self.cfg.Loss.sw * smooth_loss

class UpdateBlock(Block):

    def __init__(self, cfg, nclass):
        super().__init__()
        self.cfg = cfg
        self.nclass = nclass

        cfg = cfg.Bu

        # fbranch
        self.frame_branch = self.create_fbranch(cfg)

        # f2a: query is action
        self.f2a_layer = self.create_cross_attention(cfg, cfg.a_dim)

        # abranch
        self.action_branch = self.create_abranch(cfg)

        # a2f: query is frame
        self.a2f_layer = self.create_cross_attention(cfg, cfg.f_dim)

    def forward(self, frame_feature, action_feature, frame_pos, action_pos):
        # a->f
        action_feature = self.f2a_layer(frame_feature, action_feature, X_pos=frame_pos, Y_pos=action_pos)

        # a branch
        action_feature = self.action_branch(action_feature, action_pos)
        action_feature, action_clogit = self.process_feature(action_feature, self.nclass+1)

        # f->a
        frame_feature = self.a2f_layer(action_feature, frame_feature, X_pos=action_pos, Y_pos=frame_pos)

        # f branch
        frame_feature = self.frame_branch(frame_feature)
        frame_feature, frame_clogit = self.process_feature(frame_feature, self.nclass)

        # save features for loss and evaluation
        self.frame_clogit = frame_clogit 
        self.action_clogit = action_clogit 
        self.f2a_attn = self.f2a_layer.attn[0]
        self.a2f_attn = self.a2f_layer.attn[0]
        self.f2a_attn_logit = self.f2a_layer.attn_logit[0].unsqueeze(0)
        self.a2f_attn_logit = self.a2f_layer.attn_logit[0].unsqueeze(0)
        return frame_feature, action_feature

    def compute_loss(self, criterion: loss.MatchCriterion, match=None):
        frame_loss = criterion.frame_loss(self.frame_clogit.squeeze(1)) 
        atk_loss = criterion.action_token_loss(match, self.action_clogit)
        f2a_loss = criterion.cross_attn_loss(match, torch.transpose(self.f2a_attn_logit, 1, 2), dim=1)
        a2f_loss = criterion.cross_attn_loss(match, self.a2f_attn_logit, dim=2)

        # temporal smoothing loss
        al = loss.smooth_loss( self.a2f_attn_logit )
        fl = loss.smooth_loss( torch.transpose(self.f2a_attn_logit, 1, 2) )
        frame_clogit = torch.transpose(self.frame_clogit, 0, 1) # f, 1, c -> 1, f, c
        l = loss.smooth_loss( frame_clogit )
        smooth_loss = al + fl + l

        return atk_loss + f2a_loss + a2f_loss + frame_loss + self.cfg.Loss.sw * smooth_loss


class UpdateBlockTDU(Block):
    """
    Update Block with Temporal Downsampling and Upsampling
    """

    def __init__(self, cfg, nclass):
        super().__init__()
        self.cfg = cfg
        self.nclass = nclass

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

        # layers for temporal downsample and upsample
        self.sf_merge = nn.Sequential(nn.Linear((cfg.hid_dim+cfg.f_dim), cfg.f_dim), nn.ReLU())


    def temporal_downsample(self, frame_feature):

        # get action segments based on predictions
        cprob = frame_feature[:, :, -self.nclass:]
        _, pred = cprob[:, 0].max(dim=-1)
        pred = utils.to_numpy(pred)
        segs = utils.parse_label(pred)

        tdu = basic.TemporalDownsampleUpsample(segs)
        tdu.to(cprob.device)

        # downsample frames to segments
        seg_feature = tdu.feature_frame2seg(frame_feature)

        # refine segment features
        seg_feature, hidden = self.seg_update(seg_feature)
        seg_feature = torch.relu(seg_feature)
        seg_feature = self.seg_combine(seg_feature) # combine forward and backward features
        seg_feature, seg_clogit = self.process_feature(seg_feature, self.nclass)

        return tdu, seg_feature, seg_clogit

    def temporal_upsample(self, tdu, seg_feature, frame_feature):

        # upsample segments to frames
        s2f = tdu.feature_seg2frame(seg_feature)
        
        # merge with original framewise features to keep low-level details
        frame_feature = self.sf_merge(torch.cat([s2f, frame_feature], dim=-1))

        return frame_feature

    def forward(self, frame_feature, action_feature, frame_pos, action_pos):
        # downsample frame features to segment features
        tdu, seg_feature, seg_clogit = self.temporal_downsample(frame_feature) # seg_feature: S, 1, H

        # f->a
        seg_center = torch.LongTensor([ int( (s.start+s.end)/2 ) for s in tdu.segs ]).to(seg_feature.device)
        seg_pos = frame_pos[seg_center]
        action_feature = self.f2a_layer(seg_feature, action_feature, X_pos=seg_pos, Y_pos=action_pos)

        # a branch
        action_feature = self.action_branch(action_feature, action_pos)
        action_feature, action_clogit = self.process_feature(action_feature, self.nclass+1)

        # a->f
        seg_feature = self.a2f_layer(action_feature, seg_feature, X_pos=action_pos, Y_pos=seg_pos)

        # upsample segment features to frame features
        frame_feature = self.temporal_upsample(tdu, seg_feature, frame_feature)

        # f branch
        frame_feature = self.frame_branch(frame_feature)
        frame_feature, frame_clogit = self.process_feature(frame_feature, self.nclass)

        # save features for loss and evaluation       
        self.frame_clogit = frame_clogit 
        self.seg_clogit = seg_clogit
        self.tdu = tdu
        self.action_clogit = action_clogit 

        self.f2a_attn_logit = self.f2a_layer.attn_logit[0].unsqueeze(0)
        self.f2a_attn = tdu.attn_seg2frame(self.f2a_layer.attn[0].transpose(2, 1)).transpose(2, 1)
        self.a2f_attn_logit = self.a2f_layer.attn_logit[0].unsqueeze(0) 
        self.a2f_attn = tdu.attn_seg2frame(self.a2f_layer.attn[0])

        return frame_feature, action_feature

    def compute_loss(self, criterion: MatchCriterion, match=None):
        frame_loss = criterion.frame_loss(self.frame_clogit.squeeze(1))
        seg_loss = criterion.frame_loss_tdu(self.seg_clogit, self.tdu)
        atk_loss = criterion.action_token_loss(match, self.action_clogit)
        f2a_loss = criterion.cross_attn_loss_tdu(match, torch.transpose(self.f2a_attn_logit, 1, 2), self.tdu, dim=1)
        a2f_loss = criterion.cross_attn_loss_tdu(match, self.a2f_attn_logit, self.tdu, dim=2)

        frame_clogit = torch.transpose(self.frame_clogit, 0, 1) 
        smooth_loss = loss.smooth_loss( frame_clogit )

        return (frame_loss + seg_loss)/ 2 + atk_loss + f2a_loss + a2f_loss + self.cfg.Loss.sw * smooth_loss




