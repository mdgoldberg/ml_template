import os

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(CODE_DIR, '..')

def get_config():
    toml_path = os.path.join(ROOT_DIR, 'config.toml')
    return toml.load(toml_path)
