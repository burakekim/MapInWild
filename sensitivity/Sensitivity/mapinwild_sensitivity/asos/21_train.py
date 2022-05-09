import warnings
import os

import torch.nn as nn
import pytorch_lightning as pl

from tlib.tlearn import lightning
from asos import settings, modules, utils


def run_training(
    lr=1e-2,
    weight_decay=1e-4,
    max_epochs=50,
    n_unet_maps=3,
    continue_training=False,
):
    """
    Sets up datamodule, model and trainer. Then runs training. Feel free to change parameters within this function
    to setup model and trainer.

    :param lr: learning rate
    :param weight_decay: weight decay
    :param max_epochs: maximum number of epochs
    :return: trainer
    """

    # set random seed
    pl.seed_everything(settings.random_seed, workers=True)

    # setup datamodule
    datamodule = settings.load_datamodule(setup_stage=False)

    # setup model
    if continue_training:
        model = utils.load_model()
        model.hparams.one_cycle_lr = False  # does not work if training is continued

        checkpoint_path = '/' + os.path.join(*(settings.checkpoint_path.split('/')[:-1]), 'last.ckpt')
        print(checkpoint_path)
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            gpus=1,
            resume_from_checkpoint=checkpoint_path,
            default_root_dir=os.path.join(settings.working_folder, 'logs'),
        )

    else:
        model = modules.Model(

            in_channels=settings.in_channels,
            n_unet_maps=n_unet_maps,
            n_classes=1,

            unet_base_channels=32,  # standard UNet has 64
            double_conv=False,  # standard UNet has True, we use False
            batch_norm=True,  # standard UNet has False, we use True
            unet_mode='bilinear',  # standard UNet has None, we use 'bilinear'
            unet_activation=nn.Tanh(),

            final_activation=nn.Sigmoid(),  # nn.Sigmoid() or nn.Softmax(dim=1)
            criterion=nn.MSELoss(),  # e.g. nn.MSELoss() or nn.BCEWithLogitsLoss()

            lr=lr,
            one_cycle_lr=True,
            weight_decay=weight_decay,

            log_on_epoch=True,
            track_accuracy=True,
        )

        # setup trainer
        callbacks = lightning.get_callbacks(
            patience=max_epochs  # 20
        )

        trainer = pl.Trainer(

            max_epochs=max_epochs,
            gpus=1,
            callbacks=callbacks,
            default_root_dir=settings.working_folder,
            #deterministic=True,  # might make system slower, but ensures reproducibility
            #log_every_n_steps=1,

            #precision=16,
            #limit_train_batches=0.1,
            #limit_val_batches=0.1,
            #limit_test_batches=0.1,
            #overfit_batches=2,
            #fast_dev_run=True,
        )

    # run training
    trainer = lightning.run_training(
        trainer=trainer,
        model=model,
        datamodule=datamodule,
        test=True,
        print_data=True,
        print_model=True,
        random_seed=settings.random_seed
    )

    warnings.warn(
        f'\n\n'
        f'!!! Folder \'version_x\' in \'lightning_logs\' must be moved to working folder:\n'
        f'!!! {settings.working_folder}\n'
        f'!!! and be renamed to \'{settings.checkpoint_path.split("/")[-3]}\' to continue with the workflow.\n'
    )

    return trainer


def tune_hyperparams(
    lrs=[1e-2, 1e-3],
    weight_decays=[0, 1e-4, 1e-3, 1e-2, 1e-1],
    max_epochss=[3, 5, 10],
):
    """
    Runs training with all possible combinations of given parameters.

    :param lrs: list of learning rates
    :param weight_decays: list of weight decays
    :param max_epochss: list of maximum number of epochs
    :return:
    """

    n_runs = len(lrs) * len(weight_decays) * len(max_epochss)
    count = 0
    for lr in lrs:
        for weight_decay in weight_decays:
            for max_epochs in max_epochss:
                count += 1
                print(f'\n\n-----------------\n Run {count} out of {n_runs}.\n-----------------\n\n')

                run_training(lr=lr, weight_decay=weight_decay, max_epochs=max_epochs)


if __name__ == '__main__':

    # choose one of the following:

    # train once with standard parameters
    run_training()

    # continue training with current model
    #run_training(continue_training=True, max_epochs=max_epochs * 2)

    # train multiple times (hyperparameter tuning)
    #tune_hyperparams()
