# Deep Extreme Cut (DEXTR)
Visit our [project page](http://www.vision.ee.ethz.ch/~cvlsegmentation/dextr) for accessing the paper, and the pre-computed results.

![DEXTR](doc/dextr.png)

This is the re-implementation of our work `Deep Extreme Cut (DEXTR)`, for object segmentation from extreme points. Only testing is available, if you would like to train use our original [PyTorch](https://github.com/scaelles/DEXTR-PyTorch) repository.

### Abstract
This paper explores the use of extreme points in an object (left-most, right-most, top, bottom pixels) as input to obtain precise object segmentation for images and videos. We do so by adding an extra channel to the image in the input of a convolutional neural network (CNN), which contains a Gaussian centered in each of the extreme points. The CNN learns to transform this information into a segmentation of an object that matches those extreme points. We demonstrate the usefulness of this approach for guided segmentation (grabcut-style), interactive segmentation, video object segmentation, and dense segmentation annotation. We show that we obtain the most precise results to date, also with less user input, in an extensive and varied selection of benchmarks and datasets.

### Installation
The code was tested with [Miniconda](https://conda.io/miniconda.html) and Python 3.6. After installing the Miniconda environment:


0. Clone the repo:
    ```Shell
    git clone https://github.com/scaelles/DEXTR-KerasTensorflow
    cd DEXTR-KerasTensorflow
    ```
 
1. Install dependencies:
    ```Shell
    conda install matplotlib opencv pillow scikit-learn scikit-image h5py
    ```
    For CPU mode:
    ```Shell
    pip install tensorflow keras
    ```
    For GPU mode (CUDA 9.0 and cuDNN 7.0 is required for the latest Tensorflow version. If you have CUDA 8.0 and cuDNN 6.0 installed, force the installation of the vesion 1.4 by using ```tensorflow-gpu==1.4```. More information [here](https://www.tensorflow.org/install/)):
    ```Shell
    pip install tensorflow-gpu keras
    ```
    
  
2. Download the model by running the script inside ```models/```:
    ```Shell
    cd models/
    chmod +x download_dextr_model.sh
    ./download_dextr_model.sh
    cd ..
    ```
    The default model is trained on PASCAL VOC Segmentation train + SBD (10582 images). To download models trained on PASCAL VOC Segmentation train or COCO, please visit our [project page](http://www.vision.ee.ethz.ch/~cvlsegmentation/dextr/#downloads), or keep scrolling till the end of this README.

3. To try the demo version of DEXTR, please run:
    ```Shell
    python demo.py
    ```
    If you have multiple GPUs, you can specify which one should be used (for example gpu with id 0):
    ```Shell
    CUDA_VISIBLE_DEVICES=0 python demo.py
    ```
If installed correctly, the result should look like this:
<p align="center"><img src="doc/github_teaser.gif" align="center" width=480 height=auto/></p>

Enjoy!!

### Pre-trained models
We provide the following DEXTR models, pre-trained on:
  * [PASCAL + SBD](https://data.vision.ee.ethz.ch/csergi/share/DEXTR/dextr_pascal-sbd.h5), trained on PASCAL VOC Segmentation train + SBD (10582 images). Achieves mIoU of 91.5% on PASCAL VOC Segmentation val.
  * [PASCAL](https://data.vision.ee.ethz.ch/csergi/share/DEXTR/dextr_pascal.h5), trained on PASCAL VOC Segmentation train (1464 images). Achieves mIoU of 90.5% on PASCAL VOC Segmentation val.
  * [COCO](https://data.vision.ee.ethz.ch/csergi/share/DEXTR/dextr_coco.h5), trained on COCO train 2014 (82783 images). Achieves mIoU of 87.8% on PASCAL VOC Segmentation val.
  
### Annotation tool
[@karan-shr](https://github.com/karan-shr) has built an annotation tool based on DEXTR, which you can find here:
```
https://github.com/karan-shr/DEXTR-AnnoTool
```

### Citation
If you use this code, please consider citing the following papers:

	@Inproceedings{Man+18,
	  Title          = {Deep Extreme Cut: From Extreme Points to Object Segmentation},
	  Author         = {K.K. Maninis and S. Caelles and J. Pont-Tuset and L. {Van Gool}},
	  Booktitle      = {Computer Vision and Pattern Recognition (CVPR)},
	  Year           = {2018}
	}

	@InProceedings{Pap+17,
	  Title          = {Extreme clicking for efficient object annotation},
	  Author         = {D.P. Papadopoulos and J. Uijlings and F. Keller and V. Ferrari},
	  Booktitle      = {ICCV},
	  Year           = {2017}
	}


We thank the authors of [PSPNet-Keras-tensorflow](https://github.com/Vladkryvoruchko/PSPNet-Keras-tensorflow) for making their Keras re-implementation of PSPNet available!

If you encounter any problems please contact us at {kmaninis, scaelles}@vision.ee.ethz.ch.
