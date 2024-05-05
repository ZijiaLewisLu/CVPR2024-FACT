import os
import sys
from ..home import get_project_base
from .evaluate import Video, Checkpoint
from .utils import to_numpy
BASE = get_project_base()

def already_finished(logdir):
    fulldir = os.path.join(BASE, logdir)
    if os.path.exists(fulldir) and os.path.exists(os.path.join(fulldir, "FINISH_PROOF")):
        return True
    else:
        return False

def resume_ckpt(cfg, logdir):
    """
    return global_step, ckpt_file
    """
    if cfg.aux.resume == "" or ( not os.path.exists(logdir) ):
        print("No resume, Train from Scratch")
        return 0, None

    elif cfg.aux.resume == "max": # auto-load the latest checkpoint

        if already_finished(logdir):
            print('----------------------------------------')
            print("Exp %s %s already finished, Skip it!" % (cfg.aux.exp, cfg.aux.runid))
            print('----------------------------------------')
            sys.exit()

        # find the latest ckpt
        ckptdir = os.path.join(logdir, 'ckpts')
        network_ckpts = os.listdir(ckptdir)
        network_ckpts = [ os.path.join(ckptdir, f) for f in network_ckpts ]
        if len(network_ckpts) == 0:
            print("No resume, Train from Scratch")
            return 0, None

        iterations = [ int(os.path.basename(f)[:-4].split("-")[-1]) for f in network_ckpts ]
        load_iteration = max(iterations)
        ckpt_file = os.path.join(ckptdir, "network.iter-%d.net" % load_iteration )
        print("Resume from", ckpt_file)
        return load_iteration, ckpt_file

    else: # resume is a path to a network ckpt
        assert os.path.exists(cfg.aux.resume)
        assert cfg.split.lower() in cfg.aux.resume.lower()

        load_iteration = os.path.basename(cfg.aux.resume)
        load_iteration = int(load_iteration.split('.')[1].split('-')[1])
        print("Resume from", cfg.aux.resume)
        return load_iteration, cfg.aux.resume

def compute_null_weight(cfg, dataset):
    """
    normalized the frequency of null class to 1/num_classes
    """
    if cfg.dataset == 'epic':
        average_trans_len = dataset.average_transcript_len
        ntoken = cfg.FACT.ntoken
        num_null = ntoken - average_trans_len
        null_weight = ntoken / (num_null * ( 301 + 98 ) / 2 )
    else:
        average_trans_len = dataset.average_transcript_len
        ntoken = cfg.FACT.ntoken
        num_null = ntoken - average_trans_len
        null_weight = ntoken / (num_null * dataset.nclasses)
    cfg.defrost()
    cfg.Loss.nullw = null_weight
    cfg.freeze()
    return cfg

def save_results(ckpt: Checkpoint, vnames: list, label_list: list, attrs_saves: list) -> list:
    videos = []
    for i in range(len(vnames)):
        video = Video(vnames[i], gt_label=to_numpy(label_list[i]), **attrs_saves[i])
        videos.append(video)
    ckpt.add_videos(videos)
    return videos