# Habitizing Diffusion Planning for Efficient and Effective Decision Making


<p align="center">
·
Paper
·
<a href="#">Code</a>
·
<a href="https://bayesbrain.github.io/">Webpage</a>
·
</p>

This repository contains the PyTorch implementation of *"Habitizing Diffusion Planning for Efficient and Effective Decision Making."*

<p align="center">
    <br>
    <img src="figures/framework.png"/>
    <br>
<p>

## 🛠️ Setup

### Prerequisites
It is recommended to set up a dedicated `conda` environment.

### Create a Conda Environment
```sh
conda create -n Habi python=3.9 mesalib glew glfw pip=23 setuptools=63.2.0 wheel=0.38.4 protobuf=3.20 -c conda-forge -y
conda activate Habi
```

### Install MuJoCo Simulator and `mujoco-py` (Important)
Follow the official installation guide [here](https://github.com/openai/mujoco-py#install-mujoco).
Alternatively, you can use the script below:

```sh
#!/bin/bash
sudo apt-get update && sudo apt-get install -y wget tar libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf cmake
sudo ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so

# Set up MuJoCo
USER_DIR=$HOME
wget -c "https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz"
mkdir -p $USER_DIR/.mujoco
cp mujoco210-linux-x86_64.tar.gz $USER_DIR/mujoco.tar.gz
rm mujoco210-linux-x86_64.tar.gz
tar -zxvf $USER_DIR/mujoco.tar.gz -C $USER_DIR/.mujoco

# Update environment variables
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$USER_DIR/.mujoco/mujoco210/bin" >> ~/.bashrc
echo "export MUJOCO_PY_MUJOCO_PATH=$USER_DIR/.mujoco/mujoco210" >> ~/.bashrc
source ~/.bashrc
```

### Install Dependencies
```sh
pip install -r requirements.txt
pip install -e .
```
Additionally, you may need to install PyTorch following the [official setup guide](https://pytorch.org/) depending on your system configuration.

## 💻 Training

### Pretraining Habitization Target
To pretrain the habitization target, run:
```sh
bash pretrain_habi_target.sh
```
This script provide training for example diffusion planners. You can modifiy the `pipelines/planner_d4rl_TASK.py` to integrate any diffusion planner of your choice.

### Training Habi and Evaluating Habi
To train the **Habi** model across different tasks and seeds, run:
```sh
bash run_habi.sh
```
This script trains `habi` across various Mujoco, Kitchen, AntMaze, and Maze2D tasks with multiple random seeds for robustness. This script will transfer the diffusion planner into the corresponding habitual behavior policy, enabling fast inference and deployment.

## 📌 Evaluation Tasks
We evaluate **Habi** across multiple environments, across different types of decision-making tasks, including locomotion, manipulation, and navigation:

- **MuJoCo**:
  - HalfCheetah-Medium-Expert-v2
  - HalfCheetah-Medium-Replay-v2
  - HalfCheetah-Medium-v2
  - Hopper-Medium-Expert-v2
  - Hopper-Medium-Replay-v2
  - Hopper-Medium-v2
  - Walker2d-Medium-Expert-v2
  - Walker2d-Medium-Replay-v2
  - Walker2d-Medium-v2

- **Kitchen**:
  - Kitchen-Partial-v0
  - Kitchen-Mixed-v0

- **AntMaze**:
  - AntMaze-Medium-Play-v2
  - AntMaze-Medium-Diverse-v2
  - AntMaze-Large-Play-v2
  - AntMaze-Large-Diverse-v2

- **Maze2D**:
  - Maze2d-Umaze-v1
  - Maze2d-Medium-v1
  - Maze2d-Large-v1

These environments follow the **D4RL** benchmark.

## 🏷️ Acknowledgements
This code is built upon the [Cleandiffuser](https://github.com/CleanDiffuserTeam/CleanDiffuser) repo for consistent evaluation. Please see the [license](LICENSE) for further details.

