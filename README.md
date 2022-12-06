# MapInWild
This repository contains the code for the paper "[MapInWild: A Remote Sensing Dataset to Answer the Question What Makes Nature Wild](https://arxiv.org/abs/2212.02265)". 

MapInWild dataset is available [here](https://dataverse.harvard.edu/dataverse/mapinwild). The [Python API](https://pydataverse.readthedocs.io/en/latest/user/basic-usage.html#download-and-save-a-dataset-to-disk) of the Harvard Dataverse can be used for [bulk actions](https://guides.dataverse.org/en/5.10.1/api/dataaccess.html). 

See the folders _segmentation_ and _sensitivity_ for the _SEMANTIC SEGMENTATION_ and the _SCENE CLASSIFICATION AND SENSITIVITY ANALYSIS_ experiments, respectively. 


A sample from MapInWild dataset is shown below. The first row: four-season Sentinel-2 patches, second row: Sentinel-1 image, ESA WorldCover map, VIIRS Nighttime Day/Night band, and World Database of Protected Areas (WDPA) annotation.

![alt text](readme_aux/555556115_.png)

The files are named after the WDPA area they contain. For example, the filename of the sample above (555556115) can be tracked back to the WDPA database: https://www.protectedplanet.net/555556115.

Batch visalualizations from MapInWild dataset.

![alt text](readme_aux/batch_grid_1.png)
![alt text](readme_aux/batch_grid_2.png)



Citation
---------------------
```

@misc{https://doi.org/10.48550/arxiv.2212.02265,
  doi = {10.48550/ARXIV.2212.02265},
  
  url = {https://arxiv.org/abs/2212.02265},
  
  author = {Ekim, Burak and Stomberg, Timo T. and Roscher, Ribana and Schmitt, Michael},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Artificial Intelligence (cs.AI), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {MapInWild: A Remote Sensing Dataset to Address the Question What Makes Nature Wild},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution Non Commercial No Derivatives 4.0 International}
}
```
