import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import random
import copy

def time_mask(feature, T, num_masks, p, replace_with_zero=False, clone=False):
    """
    T: max drop length - cfg.t
    num_masks: num drop - cfg.m
    p: max drop ratio - cfg.p

    feature: T, B, H
    """
    if clone:
        feature = feature.clone()

    len_spectro = feature.shape[0]
    
    for i in range(0, num_masks):
        t = random.randrange(0, T)
        t = min( int(p*len_spectro), t )
        t_zero = random.randrange(0, len_spectro - t)

        # avoids randrange error if values are equal and range is empty
        if (t_zero == t_zero + t): 
            return feature

        if (replace_with_zero): 
            feature[t_zero:t_zero+t] = 0
        else: 
            feature[t_zero:t_zero+t] = feature.mean()
    return feature

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

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, max_len=5000, empty=False):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.empty = empty
        self.__compute_pe__(d_model, max_len)


    def __compute_pe__(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)

        if not self.empty:
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            # pe = pe.unsqueeze(0).transpose(0, 1)

        pe = pe.unsqueeze(1) 
        self.register_buffer('pe', pe)
    
    def __str__(self):
        if self.empty:
            return f"PositionalEncoding(EMPTY)"
        else:
            return f"PositionalEncoding(Dim={self.d_model}, MaxLen={self.max_len})"

    def __repr__(self):
        return str(self)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x.dim0 = sequence length
            output: [sequence length, batch_size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        if x.size(0) > self.pe.shape[0]: 
            self.__compute_pe__(self.d_model, x.size(0)+10)
            self.pe = self.pe.to(x.device)

        return self.pe[:x.size(0), :]

class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, nchannels, dropout=0.5, layernorm=True, layernorm_eps=1e-5, ngroup=1):
        super(DilatedResidualLayer, self).__init__()
        self.dilation = dilation
        self.nchannels = nchannels
        self.dropout_rate = dropout

        self.conv_dilated = nn.Conv1d(nchannels, nchannels, 3, padding=dilation, dilation=dilation, groups=ngroup)
        self.conv_1x1 = nn.Conv1d(nchannels, nchannels, 1)
        self.dropout = nn.Dropout(dropout)

        self.use_layernorm=layernorm
        if layernorm:
            self.norm = nn.LayerNorm(nchannels, eps=layernorm_eps)
        else:
            self.norm = None

    def __str__(self):
        return f"DilatedResidualLayer(Conv(d={self.dilation},h={self.nchannels}), 1x1(h={self.nchannels}), Dropout={self.dropout_rate}, ln={self.use_layernorm})"

    def __repr__(self):
        return str(self)

    def forward(self, x, mask=None):
        """
        x: B, D, T
        """
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        if mask is not None:
            x = (x + out) * mask[:, 0:1, :]
        else:
            x = x + out

        if self.norm:
            x = x.permute(0, 2, 1) # B, T, D
            x = self.norm(x)
            x = x.permute(0, 2, 1) # B, D, T

        return x

class MSTCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_layers, dropout=0.5, dilation_factor=2, ln=True, ngroup=1, in_map=False):
        super(MSTCN, self).__init__()
        if in_map:
            self.conv_1x1 = nn.Conv1d(in_dim, hid_dim, 1)
        else:
            assert in_dim == hid_dim

        self.layers = nn.ModuleList([DilatedResidualLayer(dilation_factor ** i, hid_dim, dropout, layernorm=ln, ngroup=ngroup) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(hid_dim, out_dim, 1)

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.in_map = in_map
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.dilation_factor = dilation_factor

        self.string = f"MSTCN(h:{in_dim}->{hid_dim}x{num_layers}->{out_dim}, d={dilation_factor}, ng={ngroup}, dropout={dropout}, in_map={in_map})"

    def __str__(self):
        return self.string 

    def __repr__(self):
        return str(self)

    def forward(self, x, mask=None):
        assert mask is None

        x = x.permute([1, 2, 0]) # 1, H, T

        if self.in_map:
            out = self.conv_1x1(x)
        else:
            out = x

        for layer in self.layers:
            out = layer(out, mask)

        out = self.conv_out(out) 
        out = out.permute([2, 0, 1]) # T, 1, H 

        if mask is not None:
            out = out * mask[:, 0:1, :]

        self.output = out
        return self.output

class MSTCN2(nn.Module):
    def __init__(self, dim, num_f_maps, out_dim, num_layers, dropout=0.5, dilation_factor=2, ngroup=1, ln=False,
        in_map=True,
    ):
        super().__init__()
        assert ln == False

        self.num_layers = num_layers

        self.in_map = in_map
        if self.in_map:
            self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, 1)
        else:
            assert dim == num_f_maps

        self.conv_dilated_1 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=dilation_factor**(num_layers-1-i), dilation=dilation_factor**(num_layers-1-i), groups=ngroup)
            for i in range(num_layers)
        ))

        self.conv_dilated_2 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=dilation_factor**i, dilation=dilation_factor**i, groups=ngroup)
            for i in range(num_layers)
        ))

        self.conv_fusion = nn.ModuleList((
             nn.Conv1d(2*num_f_maps, num_f_maps, 1)
             for i in range(num_layers)
            ))

        self.dropout = nn.Dropout(dropout)
        self.conv_out = nn.Conv1d(num_f_maps, out_dim, 1)

        self.string = f"MSTCN2(h:{dim}->{num_f_maps}x{num_layers}->{out_dim}, d={dilation_factor}, ng={ngroup}, dropout={dropout}, in_map={in_map})"

    def __str__(self):
        return self.string 

    def __repr__(self):
        return str(self)

    def forward(self, x):
        x = x.permute([1, 2, 0]) # 1, H, T

        if self.in_map:
            f = self.conv_1x1_in(x)
        else:
            f = x

        for i in range(self.num_layers):
            f_in = f
            f = self.conv_fusion[i](torch.cat([self.conv_dilated_1[i](f), self.conv_dilated_2[i](f)], 1))
            f = F.relu(f)
            if i != self.num_layers - 1:
                f = self.dropout(f)
            f = f + f_in

        out = self.conv_out(f)
        out = out.permute([2, 0, 1]) # T, 1, H 
        return out

class ActionUpdate_GRU(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout=0.5, layer_norm_eps=1e-5, out_map=False):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.gru = nn.GRU(in_dim, hid_dim//2, n_layers, dropout=dropout, bidirectional=True)
        self.layernorm = nn.LayerNorm(hid_dim, eps=layer_norm_eps)
        if out_map:
            self.out_map = nn.Linear(hid_dim, out_dim)
        else:
            assert hid_dim == out_dim
            self.out_map = nn.Identity()

    def forward(self, tgt, memory,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):

        return self.real_forward(tgt)
    
    def real_forward(self, action_feature):
        output, _ = self.gru(action_feature)
        output = self.layernorm(output)
        output = self.out_map(output)
        return output

####################################
####################################

def add_positional_encoding(tensor, pos):
    if pos is None:
        return tensor
    else:
        d = pos.size(-1)
        tensor = tensor.clone()
        tensor[:, :, :d] = tensor[:, :, :d] + pos
        return tensor

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class X2Y_map(nn.Module):

    def __init__(self, x_dim, y_dim, y_outdim, head_dim, dropout=0.5, kq_pos=False):
        super(X2Y_map, self).__init__()
        self.kq_pos = kq_pos

        self.X_K = nn.Linear(x_dim, head_dim)
        self.X_V = nn.Linear(x_dim, head_dim)
        self.Y_Q = nn.Linear(y_dim, head_dim)

        self.Y_W = nn.Linear(y_dim+head_dim, y_outdim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, X_feature, Y_feature, X_pos=None, Y_pos=None, X_pad_mask=None, Y_pad_mask=None, ):
        """
        X: x, b, h
        Y: y, b, h
        """
        X = X_feature.shape[0]
        Y = Y_feature.shape[0]

        if (X_pos is not None) and self.kq_pos:
            x = add_positional_encoding(X_feature, X_pos)
            xk = self.X_K(x) 
        else:
            xk = self.X_K(X_feature) 

        xv = self.X_V(X_feature)

        if (Y_pos is not None) and self.kq_pos:
            y = add_positional_encoding(Y_feature, Y_pos)
            yq = self.Y_Q(y)
        else:
            yq = self.Y_Q(Y_feature)

        assert X_pad_mask is None and Y_pad_mask is None

        attn_logit = torch.einsum('xbd,ybd->byx', xk, yq)
        attn_logit = attn_logit / math.sqrt(xk.shape[-1])
        self.attn_logit = attn_logit
        attn = torch.softmax(attn_logit, dim=-1) # B, y, x
        # if self.drop_on_att:
        #     attn = self.dropout(attn)
        
        attn_feat = torch.einsum('byx,xbh->ybh', attn, xv)
        concat_feature = torch.cat([Y_feature, attn_feat], dim=-1)
        concat_feature = self.dropout(concat_feature)
        # if not self.drop_on_att:

        Y_feature = self.Y_W(concat_feature)

        self.attn = attn.unsqueeze(1) # B, nhead=1, X, Y

        return Y_feature

class SALayer(nn.Module):
    """
    self or cross attention
    """

    def __init__(self, q_dim, nhead, dim_feedforward=2048, kv_dim=None,
                 dropout=0.1, attn_dropout=0.1,
                 activation="relu", vpos=False):
        super().__init__()

        kv_dim = q_dim if kv_dim is None else kv_dim
        self.multihead_attn = nn.MultiheadAttention(q_dim, nhead, kdim=kv_dim, vdim=kv_dim, dropout=attn_dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(q_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, q_dim)

        self.norm1 = nn.LayerNorm(q_dim)
        self.norm2 = nn.LayerNorm(q_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.q_dim = q_dim
        self.kv_dim=kv_dim
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward

        self.use_vpos = vpos
        self.dropout_rate = (dropout, attn_dropout)

    def __str__(self) -> str:
        return f"SALayer( q({self.q_dim})xkv({self.kv_dim})->{self.q_dim}, head:{self.nhead}, ffdim:{self.dim_feedforward}, dropout:{self.dropout_rate}, vpos:{self.use_vpos} )"
    
    def __repr__(self):
        return str(self)

    def forward(self, tgt, key, value, 
            query_pos: Optional[Tensor] = None,
            key_pos: Optional[Tensor] = None,
            value_pos: Optional[Tensor] = None):
        """
        tgt : query
        memory: key and value
        """
        query=add_positional_encoding(tgt, query_pos)
        key=add_positional_encoding(key, key_pos)
        if self.use_vpos:
            value=add_positional_encoding(value, value_pos)

        tgt2, self.attn = self.multihead_attn(query, key, value, average_attn_weights=False) # attn: nhead, batch, q, k

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        return tgt 

class SCALayer(nn.Module):

    def __init__(self, action_dim, frame_dim, nhead, dim_feedforward=2048, dropout=0.1, attn_dropout=0.1,
                 activation="relu", normalize_before=False, 
                 sa_value_w_pos=False, ca_value_w_pos=False):
        """
        Self-Attention + Cross-Attention Module
        """
        super().__init__()

        self.self_attn = nn.MultiheadAttention(action_dim, nhead, dropout=attn_dropout)
        self.multihead_attn = nn.MultiheadAttention(action_dim, nhead, kdim=frame_dim, vdim=frame_dim, dropout=attn_dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(action_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, action_dim)

        self.norm1 = nn.LayerNorm(action_dim)
        self.norm2 = nn.LayerNorm(action_dim)
        self.norm3 = nn.LayerNorm(action_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        assert not normalize_before
        self.sa_value_w_pos = sa_value_w_pos
        self.ca_value_w_pos = ca_value_w_pos
        
        
        self.string = f"SCALayer( adim:{action_dim}, fdim:{frame_dim}, head:{nhead}, ffdim:{dim_feedforward}, dropout:{(dropout, attn_dropout)}, svpos:{sa_value_w_pos}, cvpos:{ca_value_w_pos} )"

    def __str__(self) -> str:
        return self.string
    
    def __repr__(self):
        return str(self)

    def forward(self, tgt, memory,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        # self attention
        q = k = add_positional_encoding(tgt, query_pos)
        if not self.sa_value_w_pos:
            tgt2, self.sa_attn = self.self_attn(q, k, tgt)
        else:
            tgt2, self.sa_attn = self.self_attn(q, k, q)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        query=add_positional_encoding(tgt, query_pos)
        value = memory
        key = add_positional_encoding(memory, pos)

        if not self.ca_value_w_pos:
            tgt2, self.ca_attn = self.multihead_attn(query, key, value)
        else:
            tgt2, self.ca_attn = self.multihead_attn(query, key, key)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class SCADecoder(nn.Module):
    """
    Self+Cross-Attention Decoder
    """

    def __init__(self, in_dim, hid_dim, out_dim, decoder_layer, num_layers, norm=None, in_map=False):
        super().__init__()
        self.in_map = in_map
        if in_map:
            self.in_linear = nn.Linear(in_dim, hid_dim)
        else:
            assert hid_dim == in_dim
        self.layers = _get_clones(decoder_layer, num_layers)
        self.out_linear = nn.Linear(hid_dim, out_dim)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, pos: Optional[Tensor] = None, query_pos: Optional[Tensor] = None):

        if self.in_map:
            output = self.in_linear(tgt)
        else:
            output = tgt

        for layer in self.layers:
            output = layer(output, memory, pos=pos, query_pos=query_pos)

        if self.norm is not None:
            output = self.norm(output)

        output = self.out_linear(output)

        return output



class SADecoder(nn.Module):
    """
    Self-Attention Decoder
    """

    def __init__(self, in_dim, hid_dim, out_dim, decoder_layer: SALayer, num_layers, norm=None, in_map=False):
        super().__init__()
        self.in_map = in_map
        if in_map:
            self.in_linear = nn.Linear(in_dim, hid_dim)
        else:
            assert in_dim == hid_dim
        self.layers = _get_clones(decoder_layer, num_layers)
        self.out_linear = nn.Linear(hid_dim, out_dim)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, pos: Optional[Tensor] = None):

        if self.in_map:
            output = self.in_linear(tgt)
        else:
            output = tgt

        for layer in self.layers:
            output = layer(output, output, output, query_pos=pos, key_pos=pos, value_pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        output = self.out_linear(output)

        return output

class TemporalDownsampleUpsample():

    def __init__(self, segs):
        self.segs = segs
        self.num_seg = len(segs)

        self.seg_label = []
        for i, seg in enumerate(segs):
            self.seg_label.extend([i]*seg.len)
        self.seg_label = torch.LongTensor(self.seg_label) #.cuda()
        self.seg_lens = torch.LongTensor([s.len for s in segs]) #.cuda()

    def cuda(self):
        self.seg_label = self.seg_label.cuda()
        self.seg_lens = self.seg_lens.cuda()

    def to(self, device):
        self.seg_label = self.seg_label.to(device)
        self.seg_lens = self.seg_lens.to(device)

    def feature_frame2seg(self, frame_feature, normalize=True):
        f, b, h = frame_feature.shape
        assert b == 1

        seg_feature = torch.zeros(self.num_seg, b, h, device=frame_feature.device)
        seg_feature.index_add_(0, self.seg_label, frame_feature)

        if normalize:
            seg_feature = seg_feature / self.seg_lens[:, None, None]

        return seg_feature

    def attn_frame2seg(self, frame_attn):
        b, f, a = frame_attn.shape
        assert b == 1

        seg_attn = torch.zeros(b, self.num_seg, a, device=frame_attn.device)
        seg_attn.index_add_(1, self.seg_label, frame_attn)

        seg_attn = seg_attn / self.seg_lens[:, None]

        return seg_attn

    def feature_seg2frame(self, seg_feature):
        """
        seg_feature : S, B, H
        """
        frame_feature = seg_feature[self.seg_label]
        return frame_feature

    def attn_seg2frame(self, seg_attn):
        """
        seg_attn : B, S, A
        """
        assert seg_attn.shape[0] == 1
        frame_attn = seg_attn[0, self.seg_label].unsqueeze(0)
        return frame_attn

def _diff(x, y):
    return (x-y).abs().max()
