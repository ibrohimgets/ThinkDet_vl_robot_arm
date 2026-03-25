# ThinkDet ROS2 Demo

This folder is now a real `ament_python` ROS2 package, not just a loose node.

## What It Does

- Subscribes to RGB, depth, and camera info topics
- Accepts a natural-language query on `/thinkdet/query`
- Runs either:
  - ThinkDet (`backend:=thinkdet`), or
  - GroundingDINO fallback (`backend:=groundingdino`)
- Publishes:
  - `/thinkdet/detections`
  - `/thinkdet/grasp_target`
  - `/thinkdet/debug_image`

The 3D target is reported in the camera frame using the median depth around the
selected box center, with a fallback search inside the box if the center pixel
is invalid.

## Why This Is Better Than The Original File

- Uses the actual ThinkDet runtime path from the repo instead of a fake `ThinkDetInference` import
- Builds as a ROS2 package
- Works with real camera topics or simulator topics via parameters
- Supports a safer baseline mode when you want the demo to stay stable

## Prerequisites

- ROS2 with `rclpy`, `cv_bridge`, `message_filters`, `vision_msgs`
- Python environment with `torch`, `torchvision`, `numpy`, `Pillow`, and OpenCV
- Local project assets under the main repo:
  - `GroundingDINO/GroundingDINO/...`
  - `InternVL3_5-1B/`
  - `thinkdet/checkpoints/...` for ThinkDet mode

## Build

From the repo root:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select thinkdet_ros2_demo
source install/setup.bash
```

If the package is launched from an install space and cannot infer the repo root,
export it explicitly:

```bash
export THINKDET_REPO_ROOT=/home/iibrohimm/project/next_step
```

## Launch

ThinkDet mode:

```bash
ros2 launch thinkdet_ros2_demo thinkdet_grasp.launch.py \
  repo_root:=/home/iibrohimm/project/next_step \
  backend:=thinkdet \
  device:=cuda:0
```

GroundingDINO-only mode:

```bash
ros2 launch thinkdet_ros2_demo thinkdet_grasp.launch.py \
  repo_root:=/home/iibrohimm/project/next_step \
  backend:=groundingdino \
  device:=cuda:0
```

Simulator or alternate camera topics:

```bash
ros2 launch thinkdet_ros2_demo thinkdet_grasp.launch.py \
  repo_root:=/home/iibrohimm/project/next_step \
  rgb_topic:=/sim/camera/rgb \
  depth_topic:=/sim/camera/depth \
  camera_info_topic:=/sim/camera/camera_info
```

Send a query:

```bash
ros2 topic pub --once /thinkdet/query std_msgs/msg/String "{data: 'tool to drink coffee from'}"
```

## Project Summary

This package connects a language-conditioned open-vocabulary detector to ROS2
RGB-D camera streams and publishes a ranked detection set, a 3D target point,
and a debug visualization for robot or simulator demos.
