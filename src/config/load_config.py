import yaml
import torch


def load_config(file_path) -> dict:
    with open(file_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {config["device"]}.')
    return config