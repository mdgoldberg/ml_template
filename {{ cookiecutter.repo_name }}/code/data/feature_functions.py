"""
This is a helper module for the make_dataset step that houses all logic related to features. This is
implemented via feature functions, which are functions that take in a key (as defined in
make_dataset.py) and return a dictionary of features (this allows one to reuse code/computation if
it leads to multiple features). Whenever a "config" is used here, it is the set of [make_dataset]
configs from config.toml.
"""

def get_feature_functions(config):
    """Returns a list of feature functions, each of which returns a dictionary of features. The
    config param is a dictionary of config options, defined in config.toml under [make_dataset]. The
    idea is you can use configs to define which feature functions should be used, or to define
    "hyperparameters" for feature functions."""
    return [
        dummy_feature_function
    ]


def dummy_feature_function(key):
    return {
        'feature': 'value',
    }
