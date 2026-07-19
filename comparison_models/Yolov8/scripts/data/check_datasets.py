import os
import yaml

datasets = ['data/raw/RDD2022', 'data/raw/CRACK500', 'data/raw/Pothole600']

for dataset in datasets:
    print(f"\n{'='*50}")
    print(f"Dataset: {dataset}")
    print('='*50)

    # Check data.yaml
    yaml_path = os.path.join(dataset, 'data.yaml')
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Classes ({config.get('nc', '?')}): {config.get('names', [])}")

    # Check folder contents
    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(dataset, split)
        if os.path.exists(split_path):
            for sub in os.listdir(split_path):
                sub_path = os.path.join(split_path, sub)
                count = len(os.listdir(sub_path))
                print(f"  {split}/{sub}: {count} files")

    # Show sample label
    label_path = os.path.join(dataset, 'train', 'labels')
    if os.path.exists(label_path):
        labels = os.listdir(label_path)
        if labels:
            sample = os.path.join(label_path, labels[0])
            with open(sample, 'r') as f:
                content = f.read().strip()
            print(f"\n  Sample label ({labels[0]}):")
            print(f"  {content[:200]}")