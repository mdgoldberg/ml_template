"""
API for making datasets:

FOR USER TO DEFINE:
-------------------

config : dict of params for the dataset (from a toml file?)
    ex: start year, end year, feature parameters, etc.

get_train_keys(config)
    defines keys that are 1-to-1 with training samples

get_output_keys(config)
    defines keys that are 1-to-1 with samples that we are interested in predicting for output

get_target(config, key)
    returns the value that we wish to predict for the given training key

feature functions in feature_functions.py (see that file for more)


DEFINED AUTOMATICALLY:
----------------------

get_training_data(config)
    gets training keys, gets features for those keys, gets targets for those keys, returns dataframe

get_output_data(config)
    gets output keys, gets features for those keys, returns dataframe (w/o target column)

[ calls out to features.py to get functions to compute features ]

"""

# TODO: abstract away pandas for handling data; what sort of API would this require?
# there should be some way to default to pandas & sklearn, but override with custom functions
# 1. define an API for each case (dealing with data for pandas, modeling for sklearn)
# 2. implement each API for pandas / sklearn in a separate module
# 3. fill in API functions by forwarding to pandas/sklearn modules, but allow overriding
# 4. if I implement alternatives to pandas/sklearn, can configure which API to use in config.toml

import click
import toml

import numpy as np
import pandas as pd

import feature_functions as features


def get_train_keys(config):
    """Returns a list-like of keys on which to train, which can be of any type (tend to be strings
    or tuples). The type of keys must match the output keys."""
    pass


def get_output_keys(config):
    """Returns a list-like of keys, which can be of any type (tend to be strings or tuples). The
    type of keys must match the training keys."""
    pass


def get_target(config, key):
    """For training samples, get the target we are trying to predict for the given key."""
    pass


def get_feature_vector(key, feature_functions):
    # TODO: parallelize calling each feature_function on the key
    feature_dicts = [feature_function(key) for feature_function in feature_functions]
    all_features_dict = {k: v for feat_dict in feature_dicts for k, v in feat_dict.items()}
    return all_features_dict


def get_training_data(config):
    train_keys = get_train_keys(config)
    feature_functions = features.get_feature_functions(config)
    # TODO: parallelize get_feature_vector and get_target over keys
    # TODO: make sure this gives the proper orientation (columns as columns, not index)
    train_features = [get_feature_vector(key, feature_functions) for key in train_keys]
    train_target = [get_target(config, key) for key in train_keys]
    return pd.DataFrame(train_features), pd.Series(train_target)


def get_output_data(config):
    output_keys = get_output_keys(config)
    feature_functions = features.get_feature_functions(config)
    # TODO: parallelize get_feature_vector over keys
    # TODO: make sure this gives the proper orientation (columns as columns, not index)
    output_features = [get_feature_vector(key, feature_functions) for key in output_keys]
    return pd.DataFrame(output_features)


@click.command()
# TODO: flags for only regenerating training/output data
def main():
    config = toml.load('config.toml').get('dataset', {})
    train_X, train_y = get_training_data(config)
    output_X = get_output_data(config)
    # TODO: finish this, write to disk based on config
    print(train_X)
    print(train_y)
    print(output_X)
    train_X


if __name__ == '__main__':
    main()
