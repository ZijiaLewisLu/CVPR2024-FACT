# FACT Dataset Setup Guide

This guide explains how to prepare new datasets and generate configuration files for training FACT models.

## Dataset Requirements

Your dataset should be organized with the following structure:

```
your_dataset/
├── features/              # Video features (.npy files) [Optional]
│   ├── video1.npy
│   ├── video2.npy
│   └── ...
├── groundTruth/          # Frame-level action labels (.txt files) [Required]
│   ├── video1.txt
│   ├── video2.txt
│   └── ...
├── mapping.txt           # Action class mapping file [Required]
└── splits/              # Train/test splits [Optional]
    ├── train.split1.bundle
    ├── test.split1.bundle
    └── ...
```

### Required Files

1. **`groundTruth/` folder**: Contains frame-level labels for each video
2. **`mapping.txt`**: Maps action class IDs to action names
3. **`features/` folder**: Pre-computed video features
4. **`splits/` folder**: Predefined train/test splits

## File Formats

### `mapping.txt`
Maps action class indices to human-readable names:
```
0 background
1 take_bowl
2 crack_egg
3 pour_milk
4 stir_mixture
...
```

### `groundTruth/*.txt`
Each file contains frame-level labels, one label per line:
```
background
background
take_bowl
take_bowl
crack_egg
crack_egg
pour_milk
...
```

### `splits/*.bundle`
Lists video names for each split (without file extensions):
```
video1
video3
video5
video7
...
```

### `features/*.npy`
Pre-computed video features stored as NumPy arrays. Each file contains feature vectors for one video:
- **Shape**: `(T, D)` where T=temporal frames, D=feature dimensions
- **Example**: `video1.npy` with shape `(1500, 2048)` for a video with 1500 frames

## Create Configuration File

Once your dataset is properly organized, use the `utils/gen_config.py` script to generate a configuration file. The script helps you set dataset-related parameters, and copy other parameters from a base_config file.

### Example 

```bash
python utils/gen_config.py --dataset_path /path/to/dataset --dataset_name new_dataset --output_config new_dataset.yaml --base_config configs/breakfast.yaml
```



