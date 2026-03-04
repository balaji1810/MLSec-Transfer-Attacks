import yaml


def load_config(file_path) -> dict:
    with open(file_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config