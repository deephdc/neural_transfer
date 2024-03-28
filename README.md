---
# The Repository is ARCHIVED!
### it is now maintained in https://github.com/ai4os-hub/fast-neural-transfer/
---

DEEP Open Catalogue: Neural Transfer
==============================

[![Build Status](https://jenkins.indigo-datacloud.eu/buildStatus/icon?job=Pipeline-as-code/DEEP-OC-org/neural_transfer/master)](https://jenkins.indigo-datacloud.eu/job/Pipeline-as-code/job/DEEP-OC-org/job/neural_transfer/job/master)

**Author:** [Silke Donayre](https://github.com/SilkeDH) (KIT)

**Project:** This work is part of the [DEEP Hybrid-DataCloud](https://deep-hybrid-datacloud.eu/) project that has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 777435.

This is an example of how to perform neural transfer. This module allows you to take the content of an image and reproduce it with a new artistic style using the style of a different image. The code is based on the [Faster Neural Style Pytorch example](https://github.com/pytorch/examples/tree/master/fast_neural_style) that implements the method described in [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) developed by Justin Johnson, Alexandre Alahia and Li Fei-Fei. This module returns as a prediction either the new styled image or a pdf containing the input and the result image. You can also train a new network to have a add a new style.<br/><br/>

<p align="center">
<img src="./reports/figures/deep_examples.png" width="820">
</p>
<br/><br/>

**Table of contents**
1. [Installing this module](#installing-this-module)
    1. [Local installation](#local-installation)
    2. [Docker installation](#docker-installation)
2. [Predict](#predict)
2. [Train](#train)
3. [Acknowledgements](#acknowledgments)

## Installing this module

### Local installation

> **Requirements**

- This project has been tested in Ubuntu 18.04 with Python 3.7.4. Further package requirements are described in the `requirements.txt` file.
- It is a requirement to have [torch>=1.2.0 and torchvision>=0.5.0 installed](https://pytorch.org/get-started/locally/). 

To start using this framework clone the repo:

```bash
git clone https://github.com/deephdc/neural_transfer
cd neural_transfer
pip install -e .
```
now run DEEPaaS:
```
deepaas-run --listen-ip 0.0.0.0
```
and open http://0.0.0.0:5000/ui and look for the methods belonging to the `neural_transfer` module.

### Docker installation

We have also prepared a ready-to-use [Docker container](https://github.com/deephdc/DEEP-OC-neural_transfer) to
run this module. To run it:

```bash
docker search deephdc
docker run -ti -p 5000:5000 -p 6006:6006 -p 8888:8888 deephdc/deep-oc-neural_transfer
```

Now open http://0.0.0.0:5000/ui and look for the methods belonging to the `neural_transfer` module.


You can find more information about it in the [DEEP Marketplace](https://marketplace.deep-hybrid-datacloud.eu/modules/deep-oc-neural_transfer.html).

## Predict

Go to http://0.0.0.0:5000/ui and look for the `PREDICT` POST method. Click on 'Try it out', choose the image you want to stylize, write the name of the available style (we provide four styles: 'mosaic', 'udnie', 'rain_princess' and 'candy', you can also add more styles with the train function), choose the output type and click 'Execute'.

## Train

If you want to add a new style go to http://0.0.0.0:5000/ui and look for the `TRAIN` POST method. You will need a set of images that will be used to train the network and also the image with the new style.

There are two ways. The first one is by putting the images direct into the project folder. To do this, you need to place the following data in the following folders:

```
neural_transfer
    └──  data                     <- Put your style image here.             
         └── raw 
             └── training_dataset <- Create the training_dataset folder and put the training images inside.
         
```

The second way is to download the files that are stored remotely. To do this, you need to create a new folder in Nextcloud called neural_transfer with the following architecture:

```
neural_transfer
    ├── styles                 <- Folder containing the images with different styles to be transferred.
    ├── models                 <- Folder containing the new trained models with the learned styles.
    └── dataset                <- Folder containing the folder with the training images.
        └── training_dataset        <- Folder containing the images used for training.
```

Once you have the style and the training images allocated click on 'Try it out'. The name of the model must have the same name as the style image. For example, if your style image is named: "starry_night.jpg", the parameter `model_name` must be "starry_night.jpg" too. Then, modify the parameters if necessary and click 'Execute'. Once the training is finished you can go to the Predict section, write the name of the model without the type extension (e.g. .jpg) and stylize an image with your new style.


## Acknowledgements

The original code, etc. were created by Pytorch and can be found [here](https://github.com/pytorch/examples/tree/master/fast_neural_style).

If you consider this project to be useful, please consider citing the DEEP Hybrid DataCloud project:

> García, Álvaro López, et al. [A Cloud-Based Framework for Machine Learning Workloads and Applications.](https://ieeexplore.ieee.org/abstract/document/8950411/authors) IEEE Access 8 (2020): 18681-18692. 

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
