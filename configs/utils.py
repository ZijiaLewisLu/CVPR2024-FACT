from yacs.config import CfgNode
from .default import get_cfg_defaults
import os

def _cfg2flatdict_helper(cfg: CfgNode) -> dict:
    D = {}
    for k, v in cfg.items():
        # k = capitalize(k)
        if not isinstance(v, CfgNode):
            D[k] = v
        else:
            d = _cfg2flatdict_helper(v)
            d = { "%s.%s" % (k, k2): v for k2, v in d.items() }
            D.update(d)
            
    return D

def type_convert_helper(x):
    import torch
    t = type(x)
    if t in [int, float, bool, str, torch.Tensor]:
        return x
    else:
        return str(x)

def cfg2flatdict(cfg: CfgNode, type_convert=True) -> dict:
    D = {}
    for k, v in cfg.items():
        if not isinstance(v, CfgNode):
            D[k] = v
        else:
            d = _cfg2flatdict_helper(v)
            d = { "%s.%s" % (k, k2): v for k2, v in d.items() }
            D.update(d)

    if type_convert:
        D = { k:type_convert_helper(v) for k, v in D.items() }
            
    return D


def generate_diff_dict(default: CfgNode, cfg: CfgNode, include_missing=False) -> dict :
    """
    include_missing = False
        if a key is missing in cfg,
        it assumes the value matches with that of default
    """

    diff = {}
    for k, v in cfg.items():
        if k not in default and (not include_missing):
            continue
        if isinstance(v, CfgNode):
            subdiff = generate_diff_dict(default[k], cfg[k], include_missing=include_missing)
            if len(subdiff) > 0:
                diff[k] = subdiff
        else:
            if v != default[k]:
                diff[k] = v
    
    return diff

def capitalize(string):
    return string[0].upper() + string[1:]

def diff2expname(diff: dict, remove_leaf=False):
    string = ""
    for k, v in diff.items():
        if k.lower()  == "aux":
            continue # exclude auxiliary config
        if k.lower() == "split":
            continue # exclude split name

        if isinstance(v, dict):
            v = diff2expname(v, remove_leaf=False) # when recursive call, always false
            string += "%s[%s]-" % (k, v)
        elif not remove_leaf:
            if isinstance(v, bool):
                v = str(v)[0]
            string += "%s:%s-" % (k, v)
    
    string = string[:-1] # remove last dash
    return string


_CONFIG_FILE_DICT = {}

def generate_expname(cfg:CfgNode, cfg_file=None, default=None) -> str:
    if cfg_file is None:
        cfg_file = cfg.aux.cfg_file

    expname = []

    # add cfg_file and generate reference cfg
    if default is None:
        default = get_cfg_defaults()
    else:
        default = default.clone()

    for f in cfg_file:
        if f not in _CONFIG_FILE_DICT:
            with open(f, 'r') as fp:
                _CONFIG_FILE_DICT[f] = CfgNode.load_cfg(fp)

        default.merge_from_other_cfg(_CONFIG_FILE_DICT[f])

        f = os.path.basename(f)
        f = '.'.join(f.split('.')[:-1])
        expname.append(f)


    # add other setting
    diff = generate_diff_dict(default, cfg)
    prune = {}
    for k, v in diff.items():
        prune[capitalize(k)] = v
    diff_string = diff2expname(prune)
    if len(diff_string) > 0:
        expname.append(diff_string)
    if len(cfg.aux.mark) > 0:
        expname.append(cfg.aux.mark)

    expname = '-'.join(expname)
    return expname


def int2float_check(x, tgt):
    if isinstance(tgt, float) and "." not in x:
        try:
            int(x) # first check if x can convert to int
            x = x + '.0' # if can convert, change to float match str
        except ValueError:
            pass # cannot convert, pass on to cfg to throw error
    return x

# def hiedict2cfg(cfg_dict:dict) -> CfgNode:
#     cfg = CfgNode()
#     for k, v in cfg_dict.items():
#         if isinstance(v, dict):
#             v = hiedict2cfg(v)
#         cfg[k] = v
#     return cfg

def _get_var(c, ks: list, delete=False):
    if len(ks) == 1:
        v = c[ks[0]]
        if delete:
            del c[ks[0]]
        return v
    else:
        return _get_var(c[ks[0]], ks[1:], delete=delete)

# def _set_var(c, ks: list, v):
#     if len(ks) == 1:
#         c[ks[0]] = v
#     else:
#         # if ks[0] not in c:
#         #     c[ks[0]] = CfgNode()

#         _set_var(c[ks[0]], ks[1:], v)

# def rename_cfg_keys(cfg: CfgNode, rename_dict: dict, delete_old_key=True) -> CfgNode:

#     for old, new in rename_dict.items():
#         old = old.split('.')
#         new = new.split('.')
#         v = _get_var(cfg, old, delete=delete_old_key)
#         _set_var(cfg, new, v)

#     return cfg

def setup_cfg(cfg_file=[], set_cfgs=None, default: CfgNode=None, logdir="log/") -> CfgNode:
    """
    update default cfg according to cmd line input
    and automatic generate experiment name
    """

    if default is None:
        cfg = get_cfg_defaults()
    else:
        cfg = default.clone()

    # preprocess set_cfgs to convert int2float
    L = len(set_cfgs) if set_cfgs else 0
    new_set_cfgs = []
    for i in range(L//2):
        k = set_cfgs[i*2]
        v = set_cfgs[i*2+1]

        if not isinstance(k, list):
                k = [k]
        for k_ in k:
            tgt = _get_var(cfg, k_.split('.'))
            v_ = int2float_check(v, tgt)
            new_set_cfgs.extend([k_, v_])


    # update cfg
    for f in cfg_file: # if no config file, this is empty list
        cfg.merge_from_file(f)
    if set_cfgs is not None:
        cfg.merge_from_list(new_set_cfgs)
    cfg.aux.cfg_file = cfg_file
    cfg.aux.set_cfgs = set_cfgs

    # generate experiment name
    cfg.aux.exp = generate_expname(cfg, default=default)

    # create name of logdir
    logdir = logdir if not cfg.aux.debug else "log_test/"
    logdir = os.path.join(logdir, cfg.dataset, cfg.split,
                                    cfg.aux.exp, str(cfg.aux.runid))
    logdir = logdir.replace('-', '_') 

    cfg.aux.logdir = logdir
    return cfg


def update_from(cfg: CfgNode, ref: CfgNode, inplace=False) -> CfgNode:
    if not inplace:
        cfg = cfg.clone()
    cfg.defrost()

    for k in cfg:
        if k not in ref:
            continue

        if cfg[k] is None and ref[k] is not None:
            cfg[k] = ref[k]
    
    return cfg

