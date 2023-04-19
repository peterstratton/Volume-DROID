# Volume-DROID: All You Need is a Camera
-------------------------------------------------------------------------------------------------------------------------------
Implementation of the winter 2023 ROB 530/NA 568 team 13 final project: Volume-DROID
-------------------------------------------------------------------------------------------------------------------------------

Authors: Peter Stratton (pstratt@umich.edu), Nibarkavi Naresh Babu Amutha (nibah@umich.edu), Ashwin Saxena (ashwinsa@umich.edu), Emaad Gerami (egerami@umich.edu), Sandilya Garimella (garimell@umich.edu)

<p align="center">
  <img src="figures/output.gif" width="450">
</p>

## Overview

<p align="center">
  <img src="figures/overview.png" width="650">
</p>

Volume-DROID is a novel SLAM architecture created by combining the recent works: DROID-SLAM and NeuralBKI. Volume-DROID takes camera images (monocular or stereo) or frames from video as input and outputs online, 3D semantic mapping of the environment via combination of DROID-SLAM, point cloud registration, off-the-shelf semantic segmentation and ConvBKI. The novelty of our method lies in the fusion of DROID-SLAM and ConvBKI by the introduction of point cloud generation from RGB-Depth frames and optimized camera poses. By having only camera images or a stereo video as input, we achieved functional real-time online 3D semantic mapping.

All of our code is original, adapted from the NeuralBKI codebase, or adapted from the DROID-SLAM codebase. 

NeuralBKI code adapted from: https://github.com/UMich-CURLY/NeuralBKI \
DROID-SLAM code adapted from: https://github.com/princeton-vl/DROID-SLAM

## Installation

To install Volume-DROID, first clone the repo:
```
git clone https://github.com/peterstratton/Volume-DROID.git
```
For our experiments, all of our code was run inside Docker containers. To install Docker on an Ubuntu machine, follow the instructions in this link: https://docs.docker.com/engine/install/ubuntu/#set-up-the-repository.

After Docker is installed, we need to build a Docker image with both pytorch and ros installed. We adapted a dockerfile given to us by a classmate for this purpose. To build the image, run the following command:
```
docker build -t torch_ros - < container.Dockerfile
```

Now that the image has been built. We can run the image to launch a container using the following commmand:
```
docker run --gpus all -it --rm --net=ros --env="DISPLAY=novnc:0.0" --env="ROS_MASTER_URI=http://roscore:11311" --name pstratt_volumedroid_torch_ros -v ~/Volume-DROID:/opt/Volume-DROID/ -v ~/Volume-DROID/services.sh:/opt/Volume-DROID/services.sh --ipc=host torch_ros /opt/Volume-DROID/services.sh --shm-size 16G
```

After running the above command, you should be in the /workspace directory inside the launched container. Run the following commands to move into the Volume-DROID directory:
```
cd ../opt/Volume-DROID/
```
