

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
### Task 1 - Familiarize Yourself with the ClearnRL PPO Implementation and Training Parameters for Deep RL Learning Loop
First, check out the CleanRL PPO implementation and its configuration in [`multigrid/scripts/train_ppo_cleanrl.py`](multigrid/scripts/train_ppo_cleanrl.py). You can do this by running the following command with the `--debug-mode True` flag.

Executing this command will display the default values of the training configuration and export a video showcasing the training scenario using random actions.

Command for Task 1:
```shell
python multigrid/scripts/train_ppo_cleanrl.py --debug-mode True 
```


### Task 1 Questions
After running the above command, observe the outputs in the command line. This will provide essential information required to train your RL agent.


#### Questions for General Deep RL Training Parameters Understanding
**Q.1** From the command line outputs, can you report the following parameters? Additionally, please describe the role of each parameter in the training loop and explain how these values influence training in 1-2 sentences.

- **num_envs**: 
- **batch_size**: 
- **num_minibatches**: 
- **minibatch_size**: 
- **total_timesteps**: 
- **num_updates**: 
- **num_steps**: 
- **update_epochs**: 

**Q.2** How do these values relate to an algorithm's Sample Efficiency?

> **Note**: From Week 1, recall that `Sample Efficiency` refers to the ability of an algorithm to converge to an optimal solution with minimal sampling of experience data (trajectory from steps) from the environment.



#### Tips:
- Refer to [Part 1 of 3 — Proximal Policy Optimization Implementation: 11 Core Implementation Details](https://www.youtube.com/watch?v=MEt6rrxH8W4) from Week 2's Curriculum.
- Extensive comments and docstrings have been added atop the original [CleanRL ppo.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py) for your reference.
- Explore different configurations for the V2 environment in `CONFIGURATIONS` within [envs/__init__.py].
- Feel free to experiment with various arguments in [`multigrid/scripts/train_ppo_cleanrl.py`](multigrid/scripts/train_ppo_cleanrl.py) to familiarize yourself with this training script, its parameters, and the significance of the command line outputs.

#### Notes:
1. We only utilize the CleanRL PPO implementation in the first three main tasks of HW2. However, it offers a clean and straightforward way to grasp the ins and outs of the algorithms.
2. It's beneficial to explore other PPO implementations in CleanRL's official repository. For example:
    - [ppo_atari.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py)
    - [ppo_atari_lstm.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_lstm.py)
    - [ppo_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py)



---
## Task 2 - Understand the Dataflow in Deep RL training Loop and Implement the Technique to reduce variance in Learning 

In this task, you will delve into the specifics of the vectorized training architecture, which consists of two pivotal phases: the Rollout Phase and the Learning Phase. This is the parallelized training architecture that many Deep RL algorithms, including PPO used. You will also explore the techniques employed by PPO to reduce variance in learning, particularly focusing on the Generalized Advantage Estimation (GAE). You will enhance your understanding by identifying these phases in the code and implementing GAE to reduce variance during training in the Learning Phase when using the diversed data collected during the Rollout Phase.

### Questions to Enhance Understanding of the Deep RL Training Loop
***Q.1*** As mentioned in [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/), PPO employs a streamlined paradigm known as the vectorized architecture. This architecture encompasses two phases within the training loop:

- **Rollout Phase**: During this phase, the agent samples actions for 'N' environments and continues to process them for a designated 'M' number of steps.

- **Learning Phase**: In this phase, fundamentally, the agent learns from the data collected during the rollout phase. This data, with a length of NM, includes 'next_obs' and 'done'.

Utilizing your baseline codebase tagged `v1.1`, please pinpoint the `Rollout Phase` and the `Learning Phase` within the codebase, indicating specific line numbers. 

For instance, the lines [189-211 in CleanRL ppo.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py#L189-L211) represent the Rollout Phase in their PPO implementation.  



**Q.2 How does PPO Reduce Variance? By Utilizing Generalized Advantage Estimation (GAE)?**

> **Note**: 
  PPO employs the Generalized Advantage Estimation (GAE) method for advantage calculation, merging multiple n-step estimators into a singular estimate, thereby mitigating variance and fostering more stable and efficient training.

  GAE amalgamates multiple n-step advantage estimators into a singular weighted estimator represented as:
  
     A_t^(GAE)(gamma, lambda) = SUM(gamma*lambda)^i * delta_(t+i)

  
  where:
  - `delta_t` represents the temporal difference error defined formally as delta_t = r_t + gamma * V(s_(t+1)) - V(s_t)

  - `gamma` is the discount factor denoting the weighting of future rewards
  - `lambda` is a hyperparameter within the range [0,1], mediating the balance between bias and variance in advantage estimation

  **References**:
  "High-Dimensional Continuous Control Using Generalized Advantage Estimation" by John Schulman et al.


If you run the following training command to train an agent, you are expected to see ValErrors from blanks that needed to be filled to implement and enable Generalized Advantage Estimation (GAE). Please make use of the comments in the code to help you to implement GAE. 


Command for Task 2:
```shell
python multigrid/scripts/train_ppo_cleanrl.py --local-mode False --env-id MultiGrid-CompetitiveRedBlueDoor-v2-DTDE-Red-Single-with-Obstacle --num-envs 8 --num-gpus 0 --num-steps 128 --learning-rate 3e-4
```

#### Tips:
- Useful comments has been appended to the code for your guidance.
- For further insight, you might refer to the ["Generalized Advantage Estimation" section in "The 37 Implementation Details of Proximal Policy Optimization"](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/).

#### Notes:
- While GAE is a potent tool to mitigate variance during Deep RL training, you might explore other methodologies as well.


---

## Task 3 - How to tune the 🎲 **Exploration & Exploitation Strategies** with Algorithm Specific Hyperparamters



Command for Task 3:
```shell
python multigrid/scripts/train_ppo_cleanrl.py --local-mode False --env-id MultiGrid-CompetitiveRedBlueDoor-v2-DTDE-Red-Single-with-Obstacle --num-envs 8 --num-gpus 0 --num-steps 128 --learning-rate 3e-4
```



 (via --target-kl 0.01), 



 Report differences of training metrics





Measure sample effiicency

Wall clock time is not the same as sample efficiency 







As mentioned in 
Value Function Loss Clipping may not be importan as per Engstrom, Ilyas, et al., (2020) find no evidence that the value function loss clipping helps with the performance. Andrychowicz, et al. (2021) suggest value function loss clipping even hurts performance (decision C13, figure 43).
We implemented this detail because this work is more about high-fidelity reproduction of prior results.





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


Command for Task 4:
```shell
python multigrid/scripts/train.py --local-mode False --env MultiGrid-CompetativeRedBlueDoor-v3-DTDE-1v1 --num-workers 10 --num-gpus 0 --name 1v1_death_match --training-scheme DTDE --policies-to-train red_0  --policies-to-load blue_0 --load-dir 
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




## Task 5 - Submit your homework on Github Classroom

You can submit your results and documentations on a Jupyter Notebook or via Google CoLab Notebook. 

Please put your submission under the `submission/` folder. And you can keep your `homework1.ipynb` and related files under `notebooks/` if you are taking the notebook route.


During each training, Ray Tune will generate the MLFlow artifacts to your local directory. You will need to push your MLFlow artifacts along with your RLlib checkpoints to your submission folder in your repo.

For students not using the PRO version of Google CodeLab, 


***Note:*** 
Please beaware that the [File Size Check GitHub Action Workflow](.github/workflows/check_file_size.yml) will check the total files size for folers "submission/" "notebooks/", to ensure each of them will not exceed 5MBs. Please ensure to only submit the checkpoints, the notebooks and the MLFlow artifacts that are meant for grading by the Github Action CI/CD pipeline.