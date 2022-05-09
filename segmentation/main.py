
import os
from argparse import ArgumentParser

import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset_mapinwild_0 import MapInWild
from torch.utils.data import DataLoader

import torch
from skimage import exposure
from matplotlib import pyplot as plt
import numpy as np
from skimage import exposure
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime
import json

import torch

import segmentation_models_pytorch as smp
import pytorch_lightning as pl


class MIW(pl.LightningModule):
    def __init__(
        self,
        hparams
    ):
        super().__init__()
        self.backbone = hparams.backbone
        self.weights = hparams.weights
        self.learning_rate = hparams.lr
        self.patience = hparams.patience
        self.gpu = hparams.gpu
        self.num_workers = os.cpu_count()

        self.dataset_root = hparams.dataset_root
        self.split_file = hparams.split_file 
        self.subset_file = hparams.subset_file 
        self.bands = hparams.bands 
        self.crop_size = hparams.crop_size
        self.aux = hparams.aux
        self.batch_size = hparams.batch_size
        self.classes = hparams.classes
        
        train_transform = A.Compose([A.RandomCrop(self.crop_size[0], self.crop_size[1], p=1.0),
                            ToTensorV2(transpose_mask=True, always_apply=True, p=1.0)
                               ])

        val_transform = A.Compose([A.RandomCrop(self.crop_size[0], self.crop_size[1], p=1.0),
                            ToTensorV2(transpose_mask=True, always_apply=True, p=1.0)
                                ])
                
        test_transform = None

        self.train_dataset = MapInWild(split_file= self.split_file, root= self.dataset_root,split='train', 
                                bands= self.bands, subsetpath =  self.subset_file,
                                auxillary= self.aux, transforms=train_transform, classes=self.classes)

        self.val_dataset = MapInWild(split_file= self.split_file, root= self.dataset_root,split='validation', 
                                bands= self.bands, subsetpath =  self.subset_file, 
                                auxillary= self.aux, transforms=val_transform, classes=self.classes)
        
        self.test_dataset = MapInWild(split_file= self.split_file, root= self.dataset_root,split='test', 
                                bands= self.bands, subsetpath =  self.subset_file, 
                                auxillary= self.aux, transforms=test_transform, classes=self.classes)

        self.model = self._prepare_model()
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, image: torch.Tensor):
        return self.model(image)

    def shared_step(self, batch, stage):

        image = batch[0]
        mask = batch[1].long()

        assert image.ndim == 4

        h, w = image.shape[2:]
        
        assert h % 32 == 0 and w % 32 == 0
        assert mask.ndim == 4
        assert mask.max() <= 1.0 and mask.min() >= 0

        if self.gpu:
            image, mask = image.cuda(non_blocking=True), mask.cuda(non_blocking=True)

        logits_mask = self.forward(image)
        
        loss = self.loss_fn(logits_mask, mask)

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }


    def shared_epoch_end(self, outputs, stage):

        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")

        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_acc": accuracy,
            f"{stage}_f1": f1_score,

        }
        
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):

        self.model.train()
        torch.set_grad_enabled(True)
        if self.gpu:
            batch[0], batch[1] = batch[0].cuda(non_blocking=True), batch[1].cuda(non_blocking=True)


        return self.shared_step(batch, "train")            

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):

        self.model.eval()
        torch.set_grad_enabled(False) 

        if self.gpu:
            batch[0], batch[1] = batch[0].cuda(non_blocking=True), batch[1].cuda(non_blocking=True)

        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")  

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def train_dataloader(self):
        # DataLoader class for training
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        # DataLoader class for validation
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        # DataLoader class for validation
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def configure_optimizers(self):
        print("self.learning_rate",self.learning_rate)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)#
        return [opt], [sch]


    def _prepare_model(self):
        print("self.backbone",self.backbone)
        print("self.bands",self.bands)
        print("len_numbands",len(self.bands))

        unet_model = smp.Unet(
            encoder_name=self.backbone,
            encoder_weights=self.weights,
            in_channels=len(self.bands), 
            classes=1,
        )

        if self.gpu:
            unet_model.cuda()

        return unet_model

def main(hparams):
    
    miw_model = MIW(hparams=hparams)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="train_dataset_iou", mode="max", verbose=True
    )
    early_stopping_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor="train_dataset_iou",
        patience=(miw_model.patience * 4),
        mode="max",
        verbose=True,
    )

    logger = TensorBoardLogger(save_dir=os.getcwd(), version=1, name="lightning_logs")

    trainer = pl.Trainer(gpus=1,
                        callbacks=[checkpoint_callback, logger, early_stopping_callback],
                        precision=16, 
                        accelerator="gpu", 
                        auto_lr_find=True)
    trainer.fit(
        miw_model, 
    )

    valid_metrics = trainer.validate(miw_model, dataloaders=miw_model.val_dataloader(), verbose=False)
    print(valid_metrics)
    test_metrics = trainer.test(miw_model, dataloaders=miw_model.test_dataloader(), verbose=False) 
    print(test_metrics)
    
    logs = valid_metrics + test_metrics
     
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%H_%M_%S_%f_%b_%d_%Y")

    log_file = open("{}.json".format(timestampStr), "w")
    json.dump(logs, log_file)
    log_file.close()
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_root", default= '/data/')
    parser.add_argument("--split_file", default= '/data/aux_/split_IDs/tvt_split.csv')
    parser.add_argument("--subset_file", default= '/data/aux_/single_temporal_subset/single_temporal_subset.csv')
    parser.add_argument("--bands", default=("B4","B3","B2"))
    parser.add_argument("--crop_size", default=(512,512)) 
    parser.add_argument("--aux", default=False)
    parser.add_argument("--batch_size", default=32)
    parser.add_argument("--classes", default=['background','protected_area'])
    parser.add_argument("--backbone", default='timm-resnest14d')
    parser.add_argument("--weights", default='imagenet')
    parser.add_argument("--lr", default=0.0001)
    parser.add_argument("--patience", default=2)
    parser.add_argument("--gpu", default=True)
    args = parser.parse_args()

    main(args)