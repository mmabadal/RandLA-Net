[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/191111236/semantic-segmentation-on-semantic3d)](https://paperswithcode.com/sota/semantic-segmentation-on-semantic3d?p=191111236)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/191111236/3d-semantic-segmentation-on-semantickitti)](https://paperswithcode.com/sota/3d-semantic-segmentation-on-semantickitti?p=191111236)
[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

# RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds (CVPR 2020)

This is the official implementation of **RandLA-Net** (CVPR2020, Oral presentation), a simple and efficient neural architecture for semantic segmentation of large-scale 3D point clouds. For technical details, please refer to:
 
**RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds** <br />
[Qingyong Hu](https://www.cs.ox.ac.uk/people/qingyong.hu/), [Bo Yang*](https://yang7879.github.io/), [Linhai Xie](https://www.cs.ox.ac.uk/people/linhai.xie/), [Stefano Rosa](https://www.cs.ox.ac.uk/people/stefano.rosa/), [Yulan Guo](http://yulanguo.me/), [Zhihua Wang](https://www.cs.ox.ac.uk/people/zhihua.wang/), [Niki Trigoni](https://www.cs.ox.ac.uk/people/niki.trigoni/), [Andrew Markham](https://www.cs.ox.ac.uk/people/andrew.markham/). <br />
**[[Paper](https://arxiv.org/abs/1911.11236)] [[Video](https://youtu.be/Ar3eY_lwzMk)] [[Blog](https://zhuanlan.zhihu.com/p/105433460)] [[Project page](http://randla-net.cs.ox.ac.uk/)]** <br />
 
 
<p align="center"> <img src="http://randla-net.cs.ox.ac.uk/imgs/Fig3.png" width="100%"> </p>


	
### (1) Setup
This code has been tested with Python 3.5, Tensorflow 1.11, CUDA 9.0 and cuDNN 7.4.1 on Ubuntu 16.04.

INSTALL FROM: https://medium.com/repro-repo/install-cuda-and-cudnn-for-tensorflow-gpu-on-ubuntu-79306e4ac04e
 
- Clone the repository 
```
git clone https://github.com/mmabadal/RandLA-Net
```
- Setup python environment

---- ANACONDA ---- 

download anaconda (end of page): https://www.anaconda.com/products/individual#linux

install conda dependencies and anaconda (https://docs.anaconda.com/anaconda/install/linux/)

$ sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
$ bash ~/Downloads/Anaconda3-2020.02-Linux-x86_64.sh

you can deactivate the auto base environment on new shell
$ conda config --set auto_activate_base false

create conda env

conda create --name randlanet python=3.5 

conda activate randlanet

```
pip install -r helper_requirements.txt
pip install tensorflow-gpu==1.11
sh compile_op.sh
```

### (2) Troubleshooting

```
Ubuntu 16.04
Nvidia drivers: 384.130
CUDA: 9.0.176
CUDNN: 7.4.1
CONDA -> python:3.5
TENSORFLOW: pip installation inside CONDA (pip install tensorflow-gpu==1.11)
```

Reduced GPU memory usage: batch_size=3, numpoints=xxxx


### (3) Data structure

```
 data
 
   -classes.txt
   
   -txt
   
     -training
       -case_x
         -case_x.txt
		-annotations
			-class_x.txt
			-class_y.txt
	   -...
       
     -validation
       -case_x
         -case_x.txt
		-annotations
			-class_x.txt
			-class_y.txt
	   -...

	 -test
	   -test1
		-case_x
			-case_x.txt
			-annotations
				-class_x.txt
				-class_y.txt
		-...
	   -test2
	   	-...
 ```

### (4) Segmentation

- Preparing the dataset:
```
python utils/data_prepare.py --path_in Desktop/data/txt/ --path_out Desktop/data/ --path_cls Desktop/data/classes.txt
```
- Train:
```
python train.py --gpu 0 --data_path Desktop/data/ --path_cls Desktop/data/classes.txt
```
- Test:
```
python test.py --gpu 0 --data_path Desktop/data/  --path_cls Desktop/data/classes.txt --run RUNS/test/ --snap 25501

```

### Citation
If you find our work useful in your research, please consider citing:

	@article{hu2019randla,
	  title={RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds},
	  author={Hu, Qingyong and Yang, Bo and Xie, Linhai and Rosa, Stefano and Guo, Yulan and Wang, Zhihua and Trigoni, Niki and Markham, Andrew},
	  journal={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
	  year={2020}
	}


### Acknowledgment
-  Part of our code refers to <a href="https://github.com/jlblancoc/nanoflann">nanoflann</a> library and the the recent work <a href="https://github.com/HuguesTHOMAS/KPConv">KPConv</a>.
-  We use <a href="https://www.blender.org/">blender</a> to make the video demo.


### License
Licensed under the CC BY-NC-SA 4.0 license, see [LICENSE](./LICENSE).

