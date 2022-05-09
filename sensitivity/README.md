# MapInWild

This repository contains Jupyter notebooks to run Activation Space Occlusion Sensitivity presented in the research article "Exploring Wilderness Using Explainable Machine Learning in Satellite Imagery" (2022) by Timo T. Stomberg, Taylor Stone, Johannes Leonhardt, Immanuel Weber, and Ribana Roscher (https://doi.org/10.48550/arXiv.2203.00379) on the MapInWild dataset.

To run this repository, you must install the following git repository, running:

 - **pip install git+https://gitlab.jsc.fz-juelich.de/kiste/wilderness@main**
 
You don't need to run this, if you setup an environment using environment.yml, requirements.txt, or the setup.sh file as described below.

In the folder "asos", you find a **settings.py** file. Please change the folder directories within this file according to the location of the dataset. Here, you should also define the working_folder. This is the location where files are stored and read from the project code.


## Setup and Requirements

On a Ubuntu system and using Anaconda, you can easily use the setup.sh file to set up a virtual environment and to install the requirements automatically. Please run the following command if this repository is your current working directory:

 - **source setup.sh**

Enter "1) setup virtual environment mapinwild" to set up an virtual environment and install all requirements. Next time you can enter "2) activate virtual environment mapinwild" which will just activate the virtual environment but not install the requirements again.

You can also find the requirements in the **requirements.txt** file.

Before you run setup.sh the first time, make sure that you have installed Anaconda and the following packages or run the following lines:

 - sudo apt install python3.8-venv python3-pip
 - sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
 - sudo apt update
 - sudo apt upgrade

