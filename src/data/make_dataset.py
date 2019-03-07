# -*- coding: utf-8 -*-
from __future__ import print_function

import click
from src.logging import log_utils
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

log = log_utils.logger(__name__)


@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
def main(input_dir, output_dir):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    log.info("Processing data from {} to {}".format(input_dir, output_dir))

    # Load the dataset metadata (CSV) into memory
    log.info("Loading dataset metadata")
    data_dir = input_dir
    metadata_dir = os.path.join(input_dir, 'csv')
    dataset = ShapeNetDataset(data_dir, metadata_dir)

    # Clean the dataset (follow the rules)
    log.info("Cleaning dataset")
    dataset.clean()

    # Save the cleaned dataset to output directory
    log.info("Saving cleaned dataset")
    dataset.save(output_dir)


if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
