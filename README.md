<h1 align="center"><img src="figures/ariadne_logo.png"</h1>

----

<div align="center">
<p> Ariadne+: Deep Learning-based Augmented Framework for the Instance Segmentation of Wires </p>
</div>


# Abstract

Wires are Deformable Linear Objects (DLOs) lacking of any specific shape, color and feature. 
The proposed approach uses deep learning and standard computer vision techniques aiming at their reliable and time effective instance segmentation. 
A deep convolutional neural network is employed to generate a binary mask showing where wires are present in the input image, then graph theory is applied to create the wire paths from the binary mask through an iterative approach that aims to maximize the graph coverage. In addition, the B-Spline model of each instance, useful in manipulation tasks, is provided. 
<h1 align="center"><img src="figures/ariadne_schema.png"</h1>


# Setting Up

Main dependencies:
```
python 3.8
pytorch
cuda 10.1
opencv
scikit-image
python-igraph 0.8.3
```

use conda for creating a virtual environment and name it ```ariadneplus```.

Download the [trained models](https://drive.google.com/file/d/1rwyuUeltodsZjm53q6_46a8T-dRh1pnw/view?usp=sharing) and place them inside the ```checkpoints``` folder.


# Execution

### Script (stand-alone)
The repository contains a stand-alone implementation of Ariadne+ which can be executed (with the virtual environment enabled and inside the main directory) as:
```
python run_ariadne.py --img_path path_to_test_image
```
Additionaly with ```--show``` you can enable the display of the output whereas with ```--num_segments``` you can vary the number of superpixels used through the computation.

For example:
```
python run_ariadne.py --img_path test_images/simple_0.jpg --show --num_segments 50
```

### ROS service
The repository contains also an implementation of Ariande+ as ROS Service. Just place this repository inside a ros worksapce and execute it with:
```
roslaunch ariadne_plus ariadne.launch
```

Inside the launch file you can speciy the topic of the camera used to retrieve the image samples along with the usual parameters of Ariadne+. 

For testing purposes, you can also launch a *fake* camera topic which publishes one of the test images with:
```
roslaunch ariadne_plus fake_camera_topic.launch
```

and call the Ariande+ service with (```/get_splines``` is the topic of the service specified in ```ariadne.launch```):
```
rosservice call /get_splines "{}"
```



---
### Acknowledgements/Fundings
This work was supported by the European Commission’s Horizon 2020 Framework
Programme with the project REMODEL - Robotic technologies for the manipulation of complex deformable linear objects - under grant agreement No 870133.
  
  
DeepLabV3+ implementation based on [https://github.com/VainF/DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch)




