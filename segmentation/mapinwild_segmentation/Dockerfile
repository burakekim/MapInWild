FROM nvcr.io/nvidia/pytorch:21.04-py3
RUN pip3 install albumentations==1.1.0 matplotlib==3.5.1 pandas==1.4.0 pytorch_lightning==1.5.10 rasterio==1.2.10 scikit_image==0.19.1 openpyxl
RUN pip install git+https://github.com/qubvel/segmentation_models.pytorch
RUN apt-get update && apt-get install -y tree
WORKDIR /ws
