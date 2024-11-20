<div align='center'>

# GaussianPretrain: A Simple Unified 3D Gaussian Representation for Visual Pre-training in Autonomous Driving
### [Paper](https://arxiv.org/abs/2411.12452) | Project Page[TODO]

Shaoqing Xu<sup>1</sup>, Fang Li<sup>2</sup>, Shengyin Jiang<sup>3</sup>, Ziying Song<sup>4</sup>, Li Liu<sup>2</sup>, Zhi-xin Yang<sup>1*</sup>,
<br>

<sup>1</sup>University of Macau, <sup>2</sup>BIT, <sup>3</sup>BUPT, <sup>4</sup>BJTU‌

</div>

## Introduction

💥GussianPretrain introduces 3D Gaussian Splatting technology into vision pre-training task for the first time.
Which demonstrates remarkable effectiveness and robustness, achieving significant improvements
across various 3D perception tasks, including 3D object detection, HD map reconstruction, and occupancy prediction,
with efficiency and lower memory consumption.💥

<p align="center">
  <img src="asserts/top.png" alt="pipeline" width="1000"/>
</p>

## News

- [2024-11-20] The codebase is initialed. We are diligently preparing for a clean, optimized version. Stay tuned for the complete code release, which is coming soon.. 

- [2024-11-19] The paper is publicly available on [arXiv](https://arxiv.org/abs/2411.12452).

## Overview
💥The architecture of proposed GaussianPretrain. Given multi-view images, we first extract valid mask patches using the mask generator with the LiDAR Depth Guidance strategy. Subsequently, a set of learnable 3D Gaussian anchors is generated using ray-based guidance and conceptualized as volumetric LiDAR points. Finally, the reconstruction signals of RGB, Depth, and Occupancy are decoded based on the predicted Gaussian anchor parameters.
<p align="center">
  <img src="asserts/overview.png" alt="pipeline" width="1000"/>
</p>

## Main Results
## Getting Started