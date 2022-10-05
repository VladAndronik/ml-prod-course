import yaml


def get_config(path):
    with open(path, 'r') as f:
        conf = yaml.safe_load(f)

    return conf
