# Project Galileo
##### Supervised Learning and Classifier Training for Face Recognition
&nbsp;
[![Tensorflow](https://seedroid.com/img/post/icons/128/cc.nextlabs.tensorflow.jpg)](https://nodesource.com/products/nsolid)      [![CUDA](http://www.channelpronetwork.com/sites/default/files/styles/large/public/thumbnails/news//nvidia-cuda0.jpg?itok=TgfuEHhw)](https://nodesource.com/products/nsolid)

## Description
Project Galileo is a robust, enterprise grade multi-tasking convolutional neural net based framework optimized for speed and performance. The following are its features:

  - Image Cleaning
  - Generating Embeddings
  - Classifier Training
  - RPC Exposure
  - Notifications with Twilio Integration with SMS and Whatsapp

## Inspiration
This project is greatly inspired from the Sandberg Paper called Google FaceNet, thereby harnessing Google's whitepaper implementations and applying them for real world production ready use cases and environments as a robust solution

* [Google's Facenet] - Face Embeddings whitepapers

### Technology

Oversight uses a number of open source projects to work properly:

* [Tensorflow] - A google open-source ML framework
* [Python] - awesome language we love

### Architecture Diagram
[![Architecture](https://raw.githubusercontent.com/pourabkarchaudhuri/supervised-learning-facenet-tensorflow/master/project_galileo_architecture.png)](https://nodesource.com/products/nsolid)

### Environment Setup

##### This was built on Windows 10.

These were the pre-requisities :

##### NVIDIA CUDA Toolkit
* [CUDA] - parallel computing platform and programming model developed by NVIDIA for general computing on graphical processing units (GPUs). Download and Install all the patches. During install, choose Custom and uncheck the Visual Studio Integration checkbox.

##### Download cuDNN
* [cuDNN] - The NVIDIA CUDA® Deep Neural Network library (cuDNN) is a GPU-accelerated library of primitives for deep neural networks. Create a NVIDIA developer account to download.

##### Set Path :
Add the following paths,
&nbsp;
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin
&nbsp;
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\libnvvp
&nbsp;
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin\extras\CUPTI\libx64

##### Install [Anaconda](https://www.anaconda.com/download/) with 3.6 x64

```sh
$ conda update conda
```

##### Run package installer

```sh
$ pip install -r requirements.txt
```

##### Install C/C++ Build tools

* [C/C++ Build Tools] - Custom librarires required to build C based implementations to Python runnable builds

#### Download Pre-Trained Model

* [Pretrained Model] - A trained CNN by Google whose bottleneck needs to be consumed. Put this .pb file in the *20170512-110547* folder

#### Installation

Galileo requires [Python](https://www.python.org/) 3.6+ to run.

```sh
$ git clone https://github.com/pourabkarchaudhuri/supervised-learning-facenet-tensorflow.git
$ cd supervised-learning-facenet-tensorflow
```
#### Folder Structure
Maintain separate folders for each person inside the "/result" folder with names of people as the folder names before training

###### IMPORTANT STEP 1 : Goto send_message.py and replace required Authentication Tokens for Twilio Notifications.

###### IMPORTANT STEP 2 : Download the model from the link given in this README.md and put it in training/models/


```sh
$ python train.py
```
Now wait till you get the message "RPC Running"

#### Run Flask Server
```sh
$ python training/zerorpc_client.py
```

### Todos

 - Optimize Further to increase speed
 - Implement Docker and Jenkins based deployment

License
----

Public


   [Tensorflow]: <https://www.tensorflow.org/>
   [Python]: <https://www.python.org/>
   [Google's FaceNet]: <https://arxiv.org/abs/1503.03832>
   [Anaconda]: <https://www.anaconda.com/download/>
   [CUDA]: <https://developer.nvidia.com/cuda-90-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal>
   [cuDNN]: <https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.0.5/prod/9.0_20171129/cudnn-9.0-windows10-x64-v7>
   [Pretrained Model]: <https://drive.google.com/open?id=1sOMaZYWyWJJKJkQFVf3TUTX6-1iyR-kV>
   [C/C++ Build Tools]: <https://go.microsoft.com/fwlink/?LinkId=691126>
  
