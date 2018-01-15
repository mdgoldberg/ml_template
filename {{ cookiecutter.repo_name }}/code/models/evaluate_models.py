import click
import toml

from code.models import define_models

@click.command()
def main():
    config = toml.load('config.toml').get('models', {})
    models = define_models.get_models(config)
