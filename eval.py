import torch
from tqdm import tqdm
from .configs.utils import get_cfg_defaults
from .utils.dataset import create_dataset, DataLoader
from .utils import utils
from .utils.evaluate import Checkpoint, Video
from .utils.train_tools import save_results


for dataset_name, n_splits in [
        ['gtea', 4], ['breakfast', 4], ['egoprocel', 1], ['epic-kitchens', 1]
    ]:
    print(dataset_name)
    cfg = get_cfg_defaults()
    cfg.merge_from_file(f'./src/configs/{dataset_name}.yaml')

    ckpts = []
    for split in range(1, n_splits+1):
        cfg.split = f"split{split}"
        dataset, test_dataset = create_dataset(cfg)

        if dataset_name == 'epic-kitchens':
            from .models.blocks_SepVerbNoun import FACT
            model = FACT(cfg, dataset.input_dimension)
        else:
            from .models.blocks import FACT 
            model = FACT(cfg, dataset.input_dimension, dataset.nclasses)
        weights = f'./ckpts/{dataset_name}/split{split}-weight.pth'
        weights = torch.load(weights, map_location='cpu')
        if 'frame_pe.pe' in weights:
            del weights['frame_pe.pe']
        model.load_state_dict(weights, strict=False)
        model.eval().cuda()


        ckpt = Checkpoint(-1, bg_class=([] if cfg.eval_bg else dataset.bg_class))
        loader  = DataLoader(test_dataset, 1, shuffle=False)
        for vname, batch_seq, train_label_list, eval_label in tqdm(loader):
            seq_list = [ s.cuda() for s in batch_seq ]
            train_label_list = [ s.cuda() for s in train_label_list ]
            video_saves = model(seq_list, train_label_list)
            save_results(ckpt, vname, eval_label, video_saves)

        ckpt.compute_metrics()
        ckpts.append(ckpt)

    print(utils.easy_reduce([c.metrics for c in ckpts]))
