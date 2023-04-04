# Solving Reacher Environment using DDPG Algorithm


## 1. Introduction
In this project, we work with the 
[Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher)
environment where we have a double-jointed arm that should move to target locations. 
At ant step, a reward of +0.1 is given to the agent if the arm is in the goal 
location. 

* __Goal__: Maintaining arm's position at the target location for as many time 
steps as possible.

* __Observation space__: The observation space consists of 33 variables 
corresponding to arm's position, rotation, velocity, and angular velocities. 

* __Action space__: We have four actions each with continuous domain between -1 and
+1. The actions are corresponding to the torque exerted to two joints.

* __Algorithm__: To train the agent, we used an actor-critic method called 
__Deep Deterministic Policy Gradient (DDPG)__. 



## 2. Installation

### 2.1. Environment
First, you need to download the environment according to the Operating System:
* Linux: [[Download]](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
* Mac OS: [[Download]](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
* Windows 32bit: [[Download]](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
* Windows 64bit: [[Download]](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

Then, extract the file and put it in the root directory where 
the `Navigation.ipynb` exists. 


### 2.2. Python
For this project, you need _Python 3.9.x._ Old Anaconda version could be found 
[here](https://repo.anaconda.com/archive/).


### 2.3. Dependencies
You can download the packages needed using `requirements.txt` file:

```
pip install --upgrade pip
pip install -r requirements.txt
```


## 3. Training the Agent

The training procedure is shown in `Navigation.ipynb`. Follow this notebook to 
see how to train the agent.


## 4. Visualizing the Agent
To Visualize the agent, you can use `visualize_agent.py` file.