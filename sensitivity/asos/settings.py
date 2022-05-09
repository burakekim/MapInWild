import os

import pytorch_lightning as pl

from tlib import tlearn


# set random seed
random_seed = 0
pl.seed_everything(random_seed, workers=True)

data_folder_raw = '/media/timo/My Book/data/mapinwild/raw'  # !!! PLEASE CHANGE THIS DIRECTORY TO THE LOCATION OF THE ORIGINAL MAPINWILD DATASET !!!
infos_folder = os.path.expanduser('~/data/mapinwild/infos')  # !!! PLEASE CHANGE THIS DIRECTORY TO A FOLDER THAT INCLUDES THE FOLLOWING FILES: quality_scores.csv, single_temporal_subset.csv, split_IDs.csv !!!

data_folder_tiles = os.path.expanduser('~/data/mapinwild/tiles')  # !!! AFTER YOU HAVE TILED THE DATA, PLEASE CHANGE THIS DIRECTORY TO THE LOCATION OF THE TILED MAPINWILD DATASET !!!
data_folder_tiles_main = os.path.join(data_folder_tiles)

working_folder = os.path.expanduser('~/working_folder')  # !!! PLEASE CHANGE THIS DIRECTORY TO THE LOCATION OF YOUR WORKING FOLDER - FILES WILL BE SAVED THERE !!!


# parameters

batch_size = 32
max_image_size = 2048
in_channels = 10

checkpoint_path = os.path.join(working_folder, 'logs', 'checkpoints', 'best.ckpt')
    # the folder 'version_x' in 'lightning_logs' must be moved to working_folder and be renamed to 'logs';
    # best (not last) model is taken


# functions

def load_file_infos(only_subset: bool = True):
    """
    Loads file infos in given working_folder and returns CSVCreator object.
    If you have a file called 'file_infos.csv' in your working_folder, this data is loaded.
    If no such file exists in the working_folder, a new one is created for the data folder data_folder_tiles_main.

    :return: CSVCreator object; you can get the pandas dataframe with csv.df
    """

    csv = tlearn.data.geotif.CSVCreator(
        csv_file=os.path.join(working_folder, 'file_infos.csv'),
        folder=data_folder_tiles_main,
        include_subfolders=True,
    )

    if only_subset:
        csv.df = csv.df[csv.df['subset'] == True]

    return csv


def load_datamodule(
    batch_size: int = batch_size,
    setup_stage=False,
    cutmix: float = 0.8,  # None
):
    """
    Loads a pytorch lightning datamodule for the given file_infos.

    :param batch_size: batch size
    :param setup_stage: either 'fit', 'validate', 'test' or 'predict'
    :param cutmix: number between 0 and 1; the probability with which cutmix is performed on an image; if None,
        no cutmix if performed at all
    :return: pytorch lightning datamodule
    """

    keys = load_file_infos().df.keys()
    if 'dir' not in keys or 'label' not in keys or 'dataset' not in keys:
        raise ValueError(
            f'\n\nERROR: file_infos.csv does not contain one or more of the following keys: dir, label, dataset. '
            f'Maybe, you have not put this file into your working_folder: {working_folder}. In this case, an empty one '
            f'has been created now. Replace it with a valid file_infos.csv.'
        )

    datamodule = tlearn.data.images.DataModule(

        batch_size=batch_size,
        n_workers=8,
        setup_stage=setup_stage,

        file_infos=load_file_infos().df,
        n_classes=1,
        
        # mean of 0 and std of 10000 will normalize the s2 images from 0 to 1
        means=[0] * 10,  
        stds=[10000] * 10,

        rotate=True,
        cutmix=cutmix,
        use_rasterio=True,

        rgb_channels=[2, 1, 0],
        val_range=(0, 2**10),
    )

    return datamodule
