

---
# Assignment 2: Intro to Deep RL with Single Agent Training Environments

## Due Date
- **Due Date:** Thursday, September 21, 6:00 PM

## Overview
This assignment aims to provide hands-on experience to implementing the key components of Policy Gradient (PG)  and Actor-Critic (AC) Methods. Upon completion, you'll be able to:

- Having the architectual understanding of general PG and AC algorithms and the learning proccess in a Deep RL training via hand-on implemenation in [CleanRL](https://docs.cleanrl.dev/):
  - Understand the Deep RL training Loop and Data Flow
    - Initial Agent -> Policy Rollouts -> Rollouts Data -> Policy Training -> Update Agent -> Policy Rolluts again
    - Understand the Deep Learning Factors of Deep RL Agent like batch sizes and learning rate 
  - Understanding of the 🤖 **Deep RL Agent Mechanics**:
    - The role of 💹 **Value Functions Network** aka the Actor
    - The role of  🎯 **Policy Network** aka the Critic - 
  - How to tune the 🎲 **Exploration & Exploitation Strategies** with Algorithm Specific Hyperparamters
    - Having a good understanding of what those parameters are how to tune them to modify agent behavniors without directly modifying the envinorment mechanics (rewards, observations, dones etc...)
- Explore and exploite various PG and AC algorithms within RLLib in a 🤖 🆚 🤖 scearnio 
  - Compare and contrast how different algorithms and algorithm configs perform against a pretrained agent 


The starter code for this assignment can be found [here](https://classroom.github.com/classrooms/123430433-rl2rl-deeprl/assignments/week-1-intro-to-deep-rl-and-agent-training-environments).


## Setup Instructions
Choose to run the code on either Google Colab or your local machine:
- **Local Setup**: For local execution, install the necessary Python packages by following the [INSTALLATION.md](INSTALLATION.md) guidelines.
- **Google Colab**: To run on Colab, the notebook will handle dependency installations. Try it by clicking below:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](____________)

## Recommended Steps to Get Familiar with the new Code
We recommend reading the files in the following order. For some files, you will need to fill in the sections labeled `HW2 TODO` or `HW2 FIXME`.

- [multigrid/scripts/train_ppo_cleanrl.py](multigrid/scripts/train_ppo_cleanrl.py) # New✨ [CleanRL](https://docs.cleanrl.dev/)  with high-quality single-file implementation with research-friendly features which implemented all details on how PPO works with ~400 lines of code 
- [envs/competative_red_blue_door.py](multigrid/envs/competative_red_blue_door.py) & [envs/__init__.py](multigrid/envs/__init__.py) # Take a look at the V2 versions of new `CompetativeRedBlueDoorEnvV2`
- [wrappers.py](multigrid/wrappers.py) # Take a look at the V2 versions of new wrappers
- [scripts/train.py](multigrid/scripts/train.py) & [multigrid/utils/training_utilis.py](multigrid/utils/training_utilis.py )  # Take a look at how we are loading a pre-trained checkpoint in RLlib


Look for sections marked with `HW2` to understand how your changes will be utilized. You might also find the following files relevant:

- [scripts/manual_control.py](multigrid/scripts/manual_control.py)
- [scripts/visualize.py](multigrid/scripts/visualize.py)


Depending on your chosen setup, refer to [scripts/train.py](multigrid/scripts/train.py) and [scripts/visualize.py](multigrid/scripts/visualize.py) (if running locally), or [notebooks/homework2.ipynb](notebooks/homework2.ipynb) (if running on Colab).


If you're debugging, you might want to use VSCode's debugger. If you're running on Colab, adjust the `#@params` in the `Args` class as per the command-line arguments above.


---
# Assignment Task Breakdown

## Task 0 - Own Your Assignment By Configuring Submission Settings
- Please change the `name` in [submission_config.json](submission/submission_config.json) to your name in CamelCase
- Please tag your codebase with new release v1.1 

---
## Task 1 - Familiarize Yourself with the ClearnRL PPO Implementation
Checkout the CleanRL PPO implementation and configuration in [multigrid/scripts/train_ppo_cleanrl.py](multigrid/scripts/train_ppo_cleanrl.py) by running the following command with `--debug-mode True`.


The output of this command will print the default configuration of trianing and export an video of how the training scenario looks like with random actions. 

Command for Task 1:
```shell
python multigrid/scripts/train_ppo_cleanrl.py --debug-mode True 
```

**Tips:**
- Watch the [Part 1 of 3 — Proximal Policy Optimization Implementation: 11 Core Implementation Details](https://www.youtube.com/watch?v=MEt6rrxH8W4)
- You can try different configuraitons for the V2 environemnt in `CONFIGURATIONS` in [envs/__init__.py]
- Pleae feel free to play with various arguments in [multigrid/scripts/train_ppo_cleanrl.py](multigrid/scripts/train_ppo_cleanrl.py) to get familalr with thie new training script, the parameters and the meaning of the commandline outputs.

***Notes:***

1. We only use the CleanRL implementation in the first 2 main tasks in HW2 but it is the cleanest and simplest way to learn the in and out of the algorithms
2. It is encouraged to take a look at other implementations of ppo in CleanRL's official repos. For examples:
    - [ppo_atari.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py)
    - [ppo_atari_lstm.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_lstm.py)
    - [/ppo_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py) 



**Task 1 Description:** Run the above command and take a look at the outputs on the commandline. You should find the essental informations you need for training your RL agent. 

Can you report the following numbers from commandline outputs? And would you please describe the role of them in 1-2 sentences of how these values affect learning?

num_envs
batch_size
num_minibatches
minibatch_size
num_steps
update_epochs
total_timesteps
num_updates

fixed-length trajectory segments


How are these value relate or indicate an algorithm's sample efficiency? 

Measure sample effiicency

Wall clock time is not the same as sample efficiency 


---
## Task 2 - Understand the Deep RL training Loop and Data Flow
PPO uses a simpler clipped surrogate objective, omitting the expensive second-order optimization presented in TRPO


Please identify the Rollout Phase and the Learning Phase in the code base with given line numbers of code from your tag `v1.1`

What is the role of num_updates?


In a model-free setting, if ther agent doesn't have the model to model the transition probability even P(s1), how does it lean without the model fo the world?
  - Rollout 
  - Trail and Error? Which part? Grad maximum likelihood
  - how to reduce high variance? 
    - reward to go
    - subtract baselines

With multiple vectorized training env running in parallel, what happend if one of the i-th sub-environment is done (terminated or truncated) ?

How did PPO handle long-horion games? What is fixed-length trajectory segments?

It is important to understand next_obs and next_done’s role to help transition between phases: At the end of the j
-th rollout phase, next_obs can be used to estimate the value of the final state during learning phase, and in the begining of the (j+1)
-th rollout phase, next_obs becomes the initial observation in data. Likewise, next_done tells if next_obs is actually the first observation of a new episode. This intricate design allows PPO to continue step the sub-environments, and because agent always learns from fixed-length trajectory segments after M
 steps, PPO can train the agent even if the sub-environments never terminate or truncate. This is in principal why PPO can learn in long-horizon games that last 100,000 steps (default truncation limit for Atari games in gym) in a single episode.


Rollout phase : The agent samples actions for the N
 environments and continue to step them for a fixed number of M
 steps


Learning phase: The agent in principal learns from the collected data in the rollout phase: data of length NM
, next_obs and done




Value Function Loss Clipping may not be importan as per 

Engstrom, Ilyas, et al., (2020) find no evidence that the value function loss clipping helps with the performance. Andrychowicz, et al. (2021) suggest value function loss clipping even hurts performance (decision C13, figure 43).
We implemented this detail because this work is more about high-fidelity reproduction of prior results.




If you run the following training command to train an agent with Decentalized Training Decentalized Execution (DTDE) training scheme, you are expected to see ValueErrors from blanks that needed to be filled to fix the mismatching observation and observation space issue. Make sure to handle this exception and implement the correct observation to avoid it.


Command for Task 2:
```shell
python multigrid/scripts/train.py --local-mode False --env MultiGrid-CompetativeRedBlueDoor-v3-DTDE-Red-Single --num-workers 10 --num-gpus 0 --name --training-scheme DTDE
```

**Tips:**
- You can set `--local-mode` to True and use the VSCode debugger to walk through the code for debugging.
- Check the original definition of `self.observation_space` in [agent.py](multigrid/core/agent.py) and the new requirements in `CompetativeRedBlueDoorWrapper` in [wrappers.py](multigrid/wrappers.py) to see how the observation for the agents should be defined in `MultiGrid-CompetativeRedBlueDoor-v3`. Then you will know how to match them with the observations you are generating.

For training. Your training batch size should be larger than the horizon so that you're collecting multiple rollouts when evaluating the performance of your trained policy. For example, if the horizon is 1000 and the training batch size is 5000, you'll collect approximately 5 trajectories (or more if any of them terminate early).

***Note:*** 

You might encounter a `ValueError` for mismatching observation and observation space if you run the above command. Make sure to handle this exception and implement the correct observation to avoid it.


---

## Task 3 - How to tune the 🎲 **Exploration & Exploitation Strategies** with Algorithm Specific Hyperparamters


 (via --target-kl 0.01), 


 Report differences of training metrics



Monitor and track your runs using Tensorboard with the following command:
```shell
tensorboard --logdir submission/ray_results/
```

**Tips:**
- You can filter the plots using the following filters:

```
episode_len_mean|ray/tune/episode_reward_mean|episode_reward_min|entropy|vf|loss|kl|cpu|ram
```


- To visualize a specific checkpoint, use the following command:
```shell
python multigrid/scripts/visualize.py --env MultiGrid-CompetativeRedBlueDoor-v3-DTDE-Red-Single  --num-episodes 10  --load-dir submission/ray_results/PPO/PPO_MultiGrid-CompetativeRedBlueDoor-v3-DTDE-Red-Single_XXXX/checkpoint_YYY/checkpoint-YYY --render-mode human --gif DTDE-Red-Single
```
##### Replace `XXXX` and `YYY` with the corresponding number of your checkpoint.


- If running on Colab, use the `%tensorboard` [line magic](https://ipython.readthedocs.io/en/stable/interactive/magics.html) to achieve the same; see the [notebook](notebooks/homework1.ipynb) for more details.


## Task 4 - Explore and exploite various PG and AC algorithms within RLLib in a 🤖 🆚 🤖 scearnio 

---



## Task 4 - Explore and exploite various PG and AC algorithms within RLLib in a 🤖 🆚 🤖 scearnio 

---


## Task 5 - Submit your homework on Github Classroom

You can submit your results and documentations on a Jupyter Notebook or via Google CoLab Notebook. 

Please put your submission under the `submission/` folder. And you can keep your `homework1.ipynb` and related files under `notebooks/` if you are taking the notebook route.


During each training, Ray Tune will generate the MLFlow artifacts to your local directory. You will need to push your MLFlow artifacts along with your RLlib checkpoints to your submission folder in your repo.

For students not using the PRO version of Google CodeLab, 


***Note:*** 
Please beaware that the [File Size Check GitHub Action Workflow](.github/workflows/check_file_size.yml) will check the total files size for folers "submission/" "notebooks/", to ensure each of them will not exceed 5MBs. Please ensure to only submit the checkpoints, the notebooks and the MLFlow artifacts that are meant for grading by the Github Action CI/CD pipeline.