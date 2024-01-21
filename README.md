
# ASP Karting-Challenge-Solver

  
This project is part of the <b>Autonomous Systems Practical</b> at LMU. It covers an implementation of a reinforcement learning agent using the PPO algorithm with different probability distributions to solve a karting challenge in a Unity environment. The agent learns to navigate virtual karting tracks by optimizing its racing performance. You can train the agent to make intelligent decisions based on observed states (such as position, velocity and surroundings) and rewards (e.g. for passing checkpoints). By adjusting hyperparameters or training on several environments, the robustness of the agent can be increased. 
The project also covers the use of baseline algorithms, own plotting and analyzing features, scripts to extend the training on the <b>LMU - cip-pool</b> and an onnx-Export function.

## Table of Contents

  
  

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Workflow](#workflow)
- [Contributors](#contributors)
- [Sources](#sources)

  

## Project-Structure

### Agents
Implementation of all agents you can use based on the abstract <b>Agent</b> class.

### Configs
Training configurations and hyperparameters can be adjusted here before running a training locally.
For using a Stable-Baseline algorithm, the hyperparameters have to be adjusted in own files. 

### Environments
Unity-Environment builds can be placed in that folder to use them for trainings. Keep in mind to create a UnityEnvironment in ```main.py``` when adding new environment-builds here.

### Models
When running a training, a torch-model will be created automatically when reaching highscores in a training episode. 
When using the creating a onnx-export, the model will be saved in the unity-subfolder.

### Networks
The policy network using the beta distribution and the policy network using the MultivariateNormal distribution are placed here as well as the value network, the ppo agent uses. 

### Plotter
This folder includes plotting functions which are used when running the analyzing function. Also a html-File that includes all training plots can be found here.

### Slurm
Logs for trainings on the cip-pool can be found here (user-specific). 

### Training 
The training logic is placed in ```trainer.py``` that uses the memory structure in ```memory.py```.  

### Utils
Shared functions or enums can be found in that folder.

### Root
All executable scripts, the readme and the requirements are placed in the root folder

## Installation

  

This project is developed using Python 3.9.16

Make sure you have the appropriate Python version installed before running the project.

  

1) Clone the repository:

  

```shell

git clone https://gitlab.lrz.de/mobile-ifi/asp/SoSe23/group-3-karting-challenge-solver/

```

2) Navigate to the project directory:

```shell

cd group-3-karting-challenge-solver

```

3) Create a virtual environment (e.g. conda or venv) with the correct python version

4) Install the dependencies:

```shell

pip install -r requirements.txt

```

  
  

## Usage


  

### How to create a build of the (standard) Unity Environment

1) Open other directory

2) Clone the Karting Challenge Repository in another Folder
```shell
git clone https://gitlab.lrz.de/mobile-ifi/asp/SoSe23/karting-challenge.git
```

3) Open the Project in Unity

4) Generate a build file with the name ```karting-challenge-build```

5) Copy the build file in the ```environments``` directory in this repository
  

### Run the training (local)

1) Adjust Hyperparameters in ```configs/hyperparameters.py``` and run-config in ```configs/run_config.py``` (optional)

2) Run the training
```shell
python main.py
```


### Run the training (on cip-pool)

You can only train on the cip-pool if you are a contributor of this project because training results will automatically pushed to the repo. 

1) Connect to cip-pool via ssh 
2) Clone the repository via ssh:

3) Create a virtual environment (venv) with python version 3.8
```shell

python3 -m venv myenv
source myenv/bin/activate

```

4) Install the dependencies:

```shell

pip install -r requirements_cip.txt

```

4) Edit the name for the ``/tmp`` folder in ``launch.sh``

5) Run a training with 
```shell
SBATCH launch.sh
```

By adding any hyperparameter or run_config-parameter to this command, the configs can be adjusted for the training. For any other parameter, the standard values of the configs will be used. 
e.g.: 
```shell
SBATCH launch.sh num_episodes=300 learning_rate=0.003
```



### Analyzing

The analyzing script deletes runs that have not a minimum number of episodes (log-cleaning) and creates/updates the html-file (``plotter/training_results.html``) with the (new) training-results. 
Additionally plots for certain runs and configurations can be created here. 

```shell
python analyzing.py
```

### Export an ONNX-Model

You can export an onnx model if you adjust the <b>model</b> in ``onnx_export.py`` and running the file with:
```shell
python onnx_export.py
```

## Workflow

<b>Discord</b> was used as main communication channel. 3-4 times a week meetings were held in which project planning and occasional problems were discussed. The project progress was documented in presentations each week. Pair-Programming was also done in parts of the project.


## Contributors

- Leon Oskui: PPO-Algorithm, PPO-Beta-Distribution, Networks, HTML-Plotting and Box-Plots, Unity-Environments

- Mert Türkekul: PPO-Algorithm, PPO-Beta-Distribution, Networks, HTML-Plotting and Box-Plots, Unity-Environments

- Niklas Engel: PPO-Algorithm, Project-Structure, Logging, HTML-Plotting and Analyzing features, Slurm-Scripts, ONNX-Export

- Ozan Tanriverdi: PPO-Algorithm, Networks, Stable-Baselines, ONNX-Exports, Inference-Scripts


## Sources

### Unity ML-Agents Gym Wrapper

- https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Python-Gym-API.md

### PPO

- https://arxiv.org/pdf/1707.06347.pdf

- https://youtu.be/HR8kQMTO8bk

- https://www.youtube.com/watch?v=5P7I-xPq8u8&start=2

- https://github.com/zplizzi/pytorch-ppo/blob/master/main.py

- https://www.youtube.com/watch?v=hlv79rcHws0

### Beta Distribution

- https://proceedings.mlr.press/v70/chou17a.html

### Hyperparameter Tuning

- https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe

### Stable-Baselines3

- https://stable-baselines3.readthedocs.io/en/master/index.html

- https://pythonprogramming.net/saving-and-loading-reinforcement-learning-stable-baselines-3-tutorial/

- https://www.youtube.com/watch?v=FqNpVLKSFJg (using custom environment wrapper)

*This is a copy of the project from its original repository in LRZ GitLab*