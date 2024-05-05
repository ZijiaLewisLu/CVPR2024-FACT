#!/usr/bin/python3

import numpy as np
import argparse
import os
import json
from torch import optim
import torch
import wandb

from .utils.dataset import DataLoader, create_dataset
from .utils.evaluate import Checkpoint
from .home import get_project_base
from .configs.utils import cfg2flatdict, setup_cfg
from .utils.train_tools import resume_ckpt, compute_null_weight, save_results
from .models.loss import MatchCriterion

def evaluate(global_step, net, testloader, run, savedir):
    print("TESTING" + "~"*10)

    ckpt = Checkpoint(global_step+1, bg_class=([] if net.cfg.eval_bg else testloader.dataset.bg_class))
    net.eval()
    with torch.no_grad():
        for batch_idx, (vnames, seq_list, train_label_list, eval_label_list) in enumerate(testloader):

            seq_list = [ s.cuda() for s in seq_list ]
            train_label_list = [ s.cuda() for s in train_label_list ]
            video_saves = net(seq_list, train_label_list)
            save_results(ckpt, vnames, eval_label_list, video_saves)

    net.train()
    ckpt.compute_metrics()

    log_dict = {}
    string = ""
    for k, v in ckpt.metrics.items():
        string += "%s:%.1f, " % (k, v)
        log_dict[f'test-metric/{k}'] = v
    print(string + '\n')
    run.log(log_dict, step=global_step+1)

    fname = "%d.gz" % (global_step+1) 
    ckpt.save(os.path.join(savedir, fname))

    return ckpt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", dest="cfg_file", nargs="*",
                            help="optional config file", default=[])
    parser.add_argument("--set", dest="set_cfgs",
            help="set config keys", default=None, nargs=argparse.REMAINDER,)

    args = parser.parse_args()
    BASE = get_project_base()

    ### initialize experiment #########################################################
    cfg = setup_cfg(args.cfg_file, args.set_cfgs)
    try:
        torch.cuda.set_device('cuda:%d'%cfg.aux.gpu)
    except Exception as e:
        print(e)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.aux.gpu)

    print('============')
    print(cfg)
    print('============')

    if cfg.aux.debug:
        seed = 1 
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    logdir = os.path.join(BASE, cfg.aux.logdir)
    ckptdir = os.path.join(logdir, 'ckpts')
    savedir = os.path.join(logdir, 'saves')
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(savedir, exist_ok=True)
    print('Saving log at', logdir)

    run = wandb.init(
                project=cfg.aux.wandb_project, entity=cfg.aux.wandb_user,
                dir=cfg.aux.logdir,
                group=cfg.aux.exp, resume="allow",
                config=cfg2flatdict(cfg),
                reinit=True, save_code=False,
                mode="offline" if cfg.aux.debug else "online",
                )

    argSaveFile = os.path.join(logdir, 'args.json')
    with open(argSaveFile, 'w') as f:
        json.dump(cfg, f, indent=True)

    ### load dataset #########################################################
    dataset, test_dataset = create_dataset(cfg)
    if not cfg.aux.debug:
        trainloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    else:
        trainloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    print('Train dataset', dataset)
    print('Test dataset ', test_dataset)

    ### create network #########################################################
    if cfg.dataset == 'epic':
        from .models.blocks_SepVerbNoun import FACT
        net = FACT(cfg, dataset.input_dimension, 98, 301)
    else:
        from .models.blocks import FACT
        net = FACT(cfg, dataset.input_dimension, dataset.nclasses)

    if cfg.Loss.nullw == -1:
        compute_null_weight(cfg, dataset)
    net.mcriterion = MatchCriterion(cfg, dataset.nclasses, dataset.bg_class)

    global_step, ckpt_file = resume_ckpt(cfg, logdir)
    if ckpt_file is not None:
        ckpt = torch.load(ckpt_file, map_location="cpu")
        if 'frame_pe.pe' in ckpt: del ckpt['frame_pe.pe']
        if 'action_pe.pe' in ckpt: del ckpt['action_pe.pe']
        net.load_state_dict(ckpt, strict=False)
    net.cuda()

    print(net)

    if cfg.optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(),
                            lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(),
                            lr=cfg.lr, weight_decay=cfg.weight_decay)

    ### start training #########################################################
    start_epoch = global_step // len(trainloader)
    ckpt = Checkpoint(-1, bg_class=([] if net.cfg.eval_bg else testloader.dataset.bg_class), eval_edit=False)
    best_ckpt, best_metric = None, 0

    print(f'Start Training from Epoch {start_epoch}...')
    for eidx in range(start_epoch, cfg.epoch):

        for batch_idx, (vnames, seq_list, train_label_list, eval_label_list) in enumerate(trainloader):

            seq_list = [ s.cuda() for s in seq_list ]
            train_label_list = [ s.cuda() for s in train_label_list ]

            optimizer.zero_grad()
            loss, video_saves = net(seq_list, train_label_list, compute_loss=True)
            loss.backward()

            if cfg.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.clip_grad_norm)
            optimizer.step()

            save_results(ckpt, vnames, eval_label_list, video_saves)

            # print some progress information
            if (global_step+1) % cfg.aux.print_every == 0:

                ckpt.compute_metrics()
                ckpt.average_losses()

                log_dict = {}
                string = "Iter%d, " % (global_step+1)
                _L = len(string)
                for k, v in ckpt.loss.items():
                    log_dict[f"train-loss/{k}"] = v
                    string += f"{k}:{v:.1f}, "
                print(string)

                string = " " * _L 
                for k, v in ckpt.metrics.items():
                    string += "%s:%.3f, " % (k, v)
                    log_dict['train-metric/'+k] = v
                print(string)

                run.log(log_dict, step=global_step+1)

                ckpt = Checkpoint(-1, bg_class=(dataset.bg_class if cfg.eval_bg else []), eval_edit=False)

            # test and save model every x iterations
            if global_step != 0 and (global_step+1) % cfg.aux.eval_every == 0:
                test_ckpt = evaluate(global_step, net, testloader, run, savedir)
                if test_ckpt.metrics['F1@0.50'] >= best_metric:
                    best_ckpt = test_ckpt

                network_file = ckptdir + '/network.iter-' + str(global_step+1) + '.net'
                net.save_model(network_file)

            global_step += 1

        if cfg.lr_decay > 0 and ( eidx + 1 ) % cfg.lr_decay == 0:
            for g in optimizer.param_groups:
                g['lr'] = cfg.lr * 0.1
            print('------------------------------------Update Learning rate--------------------------------')

    print(f'Best Checkpoint: {best_ckpt.iteration}')
    best_ckpt.eval_edit = True
    best_ckpt.compute_metrics()
    best_ckpt.save(os.path.join(logdir, 'best_ckpt.gz'))
    run.finish()

    # create a file to mark this experiment has completed
    finish_proof_fname = os.path.join(logdir, "FINISH_PROOF")
    open(finish_proof_fname, "w").close()
