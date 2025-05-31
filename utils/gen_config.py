import os
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
import argparse
import yaml
from yacs.config import CfgNode as CN

# Add parent directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent))

from utils.utils import parse_label
from configs.default import get_cfg_defaults

def main():
    parser = argparse.ArgumentParser(
        description="Generate YAML configuration files for FACT model training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate config based on breakfast dataset
  python utils/gen_config.py --dataset_path /path/to/dataset --dataset_name new_dataset --output_config new_dataset.yaml --base_config configs/breakfast.yaml

Tips:
    * Use '--base_config configs/gtea.yaml' for small dataset.
    * Use '--match_mode o2m' if videos often contain many repeated actions.
        """
    )

    parser.add_argument(
        '--dataset_path', 
        type=str, 
        required=True, 
        help='Path to the dataset directory (must contain groundTruth/ folder and mapping.txt)'
    )
    parser.add_argument(
        '--dataset_name', 
        type=str, 
        default='new_dataset', 
        help='Name of the dataset (default: derived from dataset_path)'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=1, 
        help='Training batch size (default: 1)'
    )
    parser.add_argument(
        '--token_ratio', 
        type=float, 
        default=1.5, 
        help='Ratio for computing number of action tokens based on max number of action segments in a video (default: 1.5)'
    )
    parser.add_argument(
        '--ntoken', 
        type=int, 
        default=None, 
        help='Number of action tokens for the model. If specified, overrides the computed value from token_ratio (default: None)'
    )
    parser.add_argument(
        '--match_mode', 
        type=str, 
        choices=['o2o', 'o2m'], 
        default='o2o', 
        help='Token-Segment Matching mode - o2o: one-to-one, o2m: one-to-many (default: o2o)'
    )
    parser.add_argument(
        '--bg_class', 
        type=int, 
        default=0, 
        help='Background class label index (default: 0)'
    )
    parser.add_argument(
        '--base_config', 
        type=str, 
        default=None, 
        required=True,
        help='Path to base YAML config file to extend (default: None)'
    )
    parser.add_argument(
        '--output_config', 
        type=str, 
        default=None, 
        help='Output config file path (default: config_{dataset}_{mode}.yaml)'
    )
    parser.add_argument(
        '--feature_transpose', 
        action='store_true', 
        help='Set this to true if features are store in shape of HiddenDim x TemporalDim (default: False)'
    )

    args = parser.parse_args()

    # Use dataset name from args or derive from path
    dataset_name = args.dataset_name or Path(args.dataset_path).name
    
    dataset_path = Path(args.dataset_path)
    print(f"Processing dataset: {dataset_name}")
    print(f"Dataset path: {dataset_path}")

    # Validate dataset structure
    feature_folder = dataset_path / 'features'
    label_folder = dataset_path / 'groundTruth'
    mapping_file = dataset_path / 'mapping.txt'

    if not label_folder.exists():
        raise FileNotFoundError(f"Label folder {label_folder} does not exist")
    
    if not mapping_file.exists():
        raise FileNotFoundError(f"Mapping file {mapping_file} does not exist")
        
    print(f"✓ Found label folder: {label_folder}")
    print(f"✓ Found mapping file: {mapping_file}")
    
    if feature_folder.exists():
        print(f"✓ Found feature folder: {feature_folder}")
    else:
        print(f"⚠ Feature folder not found: {feature_folder}")

    # Check for splits folder (optional)
    splits_folder = dataset_path / 'splits'
    if splits_folder.exists():
        print(f"✓ Found splits folder: {splits_folder}")
    else:
        print(f"⚠ Splits folder not found: {splits_folder}")

    # Analyze labels to compute statistics
    label_files = sorted(label_folder.glob('*.txt'))
    if not label_files:
        raise FileNotFoundError(f"No label files found in {label_folder}")
    
    print(f"Found {len(label_files)} label files")
    
    lens = []
    
    for label_file in tqdm(label_files, desc="Analyzing labels"):
        with open(label_file, 'r') as f:
            labels = f.read().strip().split('\n')
        
        if args.match_mode == 'o2o':
            # For one-to-one matching, compute number of segments
            action_segs = parse_label(labels)
            lens.append(len(action_segs))
        elif args.match_mode == 'o2m':
            # For one-to-many matching, compute number of unique actions
            lens.append(len(set(labels)))

    average_len = np.mean(lens)
    max_len = np.max(lens)
    
    if args.ntoken is None:
        num_token = int(max_len * args.token_ratio)
    else:
        num_token = args.ntoken
    
    print(f"Dataset Statistics:")
    print(f"Average number of segments in video: {average_len:.2f}")
    print(f"Max number of segments in video: {max_len}")
    print(f"Computed number of tokens: {num_token}")

    # Load base config
    print(f"Loading base config from: {args.base_config}")
    with open(args.base_config, 'r') as f:
        cfg = CN.load_cfg(f)

    # Set dataset-specific parameters
    cfg.dataset = dataset_name
    cfg.batch_size = args.batch_size
    cfg.feature_path = str(feature_folder) 
    cfg.groundTruth_path = str(label_folder)
    cfg.split_path = str(splits_folder)
    cfg.map_fname = str(mapping_file)
    cfg.feature_transpose = args.feature_transpose
    cfg.bg_class = args.bg_class
    cfg.average_transcript_len = float(average_len)
    
    # Set model parameters
    cfg.FACT.ntoken = num_token
    cfg.Loss.match = args.match_mode

    # Generate output filename if not provided
    if args.output_config is None:
        output_config = f"config_{dataset_name}_{args.match_mode}.yaml"
    else:
        output_config = args.output_config
    
    # Save config
    print(f"\nSaving config to: {output_config}")
    with open(output_config, 'w') as f:
        # yaml.dump(cfg.dump(), f, default_flow_style=False, indent=2)
        f.write(cfg.dump())
    
    print(f"✓ Config file generated successfully!")
    
    print('Plrease refer to README.md for training command.')
    print("\nTip: You can customize the generated config file by adjusting:")
    print("  • Train/test split (split)")
    print("  • Feature sampling rate (sr)")
    print("  • Number of action tokens (ntoken)")
    print("  • Training epochs and Learning rate decay step (lr_decay)")

if __name__ == "__main__":
    main()




