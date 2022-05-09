import os
from typing import Callable, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from torch import Tensor
import pandas as pd

class MapInWild(torch.utils.data.Dataset): 
    
    BAND_SETS: Dict[str, Tuple[str, ...]] = {
        "all": (
            "VV",
            "VH",
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7",
            "B8",
            "B8A",
            "B11",
            "B12",
        ),        "s1": ("VV", "VH"),
        "s2-all": (
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7",
            "B8",
            "B8A",
            "B11",
            "B12",
        )
    }

    band_names = (
        "VV",
        "VH",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B11",
        "B12",
    )
        
    def __init__(
        self,
        split_file, 
        auxillary: bool = False, 
        subsetpath: bool = True, 
        root: str = "data",
        split: str = "train",
        bands: Sequence[str] = BAND_SETS["all"],
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None) -> None:
        """
        Initialize a new MapInWild dataset instance.

        Args: 
            split_file: path to the T/V/T split file
            auxillary: if False outputs the concatenated Sentinel-1 and Sentinel-2 bands; if True additionaly outputs ESA WC and VIIRS data  
            subsetpath: path to the single temporal subset file
            root: path to the dataset folder
            split: "train", "validation", or "test"
            bands: a sequence of band indices to use where the indices correspond to the
                array index of combined Sentinel 1 and Sentinel 2
            transforms:a function/transform that takes input sample and its target as
                entry and returns a transformed version
        """
        assert split in ["train","validation", "test"]

        self.band_indices = torch.tensor([self.band_names.index(b) for b in bands]).long()
        self.bands = bands

        self.root = root
        self.split = split
        self.subsetpath = subsetpath
        self.auxillary = auxillary
        self.transforms = transforms
        
        split_dataframe = pd.read_csv(split_file)

        self.ids = split_dataframe[split].dropna().values.tolist() 
        self.ids = [int(i) for i in self.ids]

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """
        Return an index within the dataset.
        """
        filename = self.ids[index]

        mask = self._load_raster(filename, "mask")

        mask[mask != 0] = 1 

        s1 = self._load_raster(filename, "S1")
        
        if not self.subsetpath:
            s2 = self._load_raster(filename, "s2_summer")
            
        if self.subsetpath:
            season = self.get_subset_s2_season(self.subsetpath, filename)
            s2 = self._load_raster(filename,str(season))

        s2 = s2/10000 
        
        image = torch.cat(tensors=[s1, s2], dim=0) 
        image = torch.index_select(image, dim=0, index=self.band_indices)

        if self.transforms is not None:
            image = np.transpose(image.numpy(), (1, 2, 0))
            mask = np.transpose(mask.numpy(), (1, 2, 0))          
  
            transformed = self.transforms(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

            image = np.transpose(image, (2, 0, 1))
            mask = np.transpose(mask, (2, 0, 1))

        if not self.auxillary:
            image = image
            mask = mask
            sample: Dict[str, Tensor] = {"image": image, "mask": mask} 

        if self.auxillary:
            esa_wc = self._load_raster(filename, "ESA_WC")
            viirs = self._load_raster(filename, "VIIRS")
            sample: Dict[str, Tensor] = {"image": image, "aux_esa_wc": esa_wc, "aux_niir":niir, "mask": mask}

        return sample
    
    def __len__(self) -> int:
        """
        Return the number of data points in the dataset.
        """
        return len(self.ids)

    def _load_raster(self, filename: str, source: str) -> Tensor:
        """
        Load a single raster image or target.
        """
        with rasterio.open(
                os.path.join(self.root,
                                "{}".format(source), 
                                "{}.tif".format(filename), 
                )
        ) as f:
            array = f.read().astype(np.int32)
            tensor: Tensor = torch.from_numpy(array)  
            return tensor

    @staticmethod
    def get_subset_s2_season(path_to_subset_seasons, data_id):
        """
        Query the single temporal subset out of four sentinel-2 images. 
        """
        subset_season_dataframe = pd.read_csv(path_to_subset_seasons)
        sts = subset_season_dataframe['single_temporal_subset'].tolist()
        im_id = subset_season_dataframe['imagePath'].tolist()
        zip_sts_id = dict(zip(im_id, sts))
        season = zip_sts_id[data_id]  
        s2_season = "s2_{}".format(season.lower())

        return s2_season

    @staticmethod
    def stretch(im):
        """
        Strectch using 1st and 99th percentile. Used for visualization.
        """
        p1, p99 = np.percentile(im, (1, 99))
        J = exposure.rescale_intensity(im, in_range=(p1, p99))
        J = J/J.max()
        return J