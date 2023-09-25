

---
# Assignment 3: Apply MARL Techniques to Solve Multi-Agent Stochastic Games  

## Due Date
- **Due Date:** Thursday, September 28, 9:00 PM

## Overview
This assignment provides a hands-on introduction to develop Multi-Agent Systems to solve Stochastic Tasks with Multi-Agent Reinforcement Learning (MARL). This excersize provide practical solution to tackle therotical concepts. It is meant to as the test bed and equipped you with tricks and tools to build the blueprint to solve challenging and complex MARL problems. By the end of this assignment, you'll be equipped to:

- Having the hands-on understanding of the three Multi-Agent Training Schemes:
  - **Decentralized Training with Decentralized Execution (DTDE)**: Independent training & execution per agent without central coordination
  - **Centralized Training with Decentralized Execution (CTDE)**: Central training for joint policies, but agents act independently in execution
  - **Centralized Training with Centralized Execution (CTCE)**: Teams of homogeneous agents share policy, rewards, and parameters
- Solving the following Game Models in both Collaborative and Competitive Tasks:
  - Collaborative Task:
    - Decentralized Partially-Observable Identical-Interest Stochastic Potential Game
      - Model as potential games if agents have different rewards, mission and objectives 
        - Agents can learn to coordinate to solve sub-tasks that contribute to the overall team objective
  - Competitive Task:
    - Decentralized Partially-Observable Zero-Sum Stochastic Games (1v1)
    - Centralized Partially-Observable General-Sum Stochastic Team Games (2v2)
- Design and Implement the following Deep RL Capabilities/Mechanics with RLlib:
  - Restore Checkpoint trained in scenario with limited world dynamics and transfer the "skills" aka NN parameters to solve scneario with randomness
  - Implement your Customized Torch PPO Policy to perform parameters sharing during trianing but independent execution during depolyment (CTDE)
  - Implement Policy Self-Play Callback Function to train your main agent with a pool of legacy versions of the main agent
- Apply Everything You Learned in the Class to Customize your best agent to compete in the STR MARL CUP (1v1)


The starter code for this assignment can be found [here](https://classroom.github.com/a/flIqv1Tb).


## Setup Instructions
Choose to run the code on either Google Colab or your local machine:
- **Local Setup**: For local execution, install the necessary Python packages by following the [INSTALLATION.md](INSTALLATION.md) guidelines.
- **Google Colab**: To run on Colab, the notebook will handle dependency installations. Try it by clicking below:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](____________)

## Recommended Steps to Get Familiar with the new Code
We recommend reading the files in the following order. For some files, you will need to fill in the sections labeled `HW2 TODO` or `HW2 FIXME`.

- [multigrid/scripts/train_ppo_cleanrl.py](multigrid/scripts/train_ppo_cleanrl.py) # New✨ [CleanRL](https://docs.cleanrl.dev/)  with high-quality single-file implementation with research-friendly features which implemented all details on how PPO works with ~400 lines of code (without additional reference comments)
- [envs/competative_red_blue_door.py](multigrid/envs/competative_red_blue_door.py) & [envs/__init__.py](multigrid/envs/__init__.py) # Take a look at the V2 versions of new `CompetativeRedBlueDoorEnvV2`
- [wrappers.py](multigrid/wrappers.py) # Take a look at the V2 versions of new wrappers
- [scripts/train.py](multigrid/scripts/train.py) & [multigrid/utils/training_utilis.py](multigrid/utils/training_utilis.py )  # Take a look at how we are loading a pre-trained checkpoint in RLlib


Look for sections marked with `HW3` to understand how your changes will be utilized. You might also find the following files relevant:

- [scripts/manual_control.py](multigrid/scripts/manual_control.py)
- [scripts/visualize.py](multigrid/scripts/visualize.py)


Depending on your chosen setup, refer to [scripts/train.py](multigrid/scripts/train.py) and [scripts/visualize.py](multigrid/scripts/visualize.py) (if running locally), or [notebooks/homework2.ipynb](notebooks/homework2.ipynb) (if running on Colab).


If you're debugging, you might want to use VSCode's debugger. If you're running on Colab, adjust the `#@params` in the `Args` class as per the command-line arguments when running locally.

If you need more compute Resource, please let me know. I can enable Github CodeSpace for limited users see Appendix for more info.

---
# Assignment Task Breakdown

## Task 0 - Own Your Assignment By Configuring Submission Settings
- Please change the `name` in [submission_config.json](submission/submission_config.json) to your name in CamelCase
- Please tag your codebase with new release v3.1 

  Git tagging command:
  ```sheel
  git tag -a v3.1 -m "Baseline 3.1"
  ```

---
## Task 1 - Enable Centralized Training with Centralized Execution(CTCE) with CentralizedCritic and TorchCentralizedCriticModel

In this task, you will complete the implementation of PPO `CentralizedCritic` to train your CTDE agents to solve a collaborative task involving three sub-tasks: grabbing the Red key, removing the Blue Ball blocking the Red door, and unlocking the Red door with the Red key.

Unlike what we have seen before, we will train our agents to generalize to the random placements of the Red key.

<figure style="text-align: center;">
    <img src="submission/evaluation_reports/from_github_actions/CTDE-Red_YourName.gif" alt="Local GIF" style="width:400px;"/>
    <figcaption style="text-align: center;">Trained CTDE agents without Randomness</figcaption>
</figure>

<figure style="text-align: center;">
    <img src="submission/evaluation_reports/from_github_actions/CTDE-Red-Eval_YourName.gif" alt="Local GIF" style="width:400px;"/>
    <figcaption style="text-align: center;">Zero-shot Testing with Randomness</figcaption>
</figure>


To complete your implementation, identify the `HW3 - TODOs` within [multigrid/rllib/ctde_torch_policy.py](multigrid/rllib/ctde_torch_policy.py) and fill in the necessary sections.

> **Heads Up!**: This implementation of CentralizedCritic and TorchCentralizedCriticModel is modified on top of RLLib's official [examples](https://github.com/ray-project/ray/blob/master/rllib/examples/centralized_critic.py). 


#### Tips:
- For more insights, take a look at RLlib's official documentation on [Implementing a Centralized Critic](https://docs.ray.io/en/latest/rllib/rllib-env.html#implementing-a-centralized-critic).


---
## Task 2 - Solving the Decentralized Partially-Observable Identical-Interest MDP with your CTDE Agents


**Reflection Question**: Why are we using Centralized Training with Decentralized Execution (CTDE)?  

<figure style="text-align: center;">
    <img src="images/HW_images/HW3/MARL_challenges_and_solutions.png" alt="Local GIF" style="width:800px;"/>
    <figcaption style="text-align: center;">MARL Challenges and Solutions Mapping Diagrams</figcaption>
</figure>

In Class, we learned a few frameworks for mapping MARL solutions to problems. For our Decentralized-POMDP collaborative task, we anticipate challenges like partial observability and non-stationarity, and possibly the Credit Assignment problem with entirely disjoint agents.

Thus, we are using CTDE. This method centrally trains a joint policy considering all agents' actions and observations. At deployment, each agent, following its own policy derived from centralized training. Recall the Dec-POMDP as a multi-agent system. Agents, with unique policies, interact in a partially observable environment, making local decisions.

Our game's objective is to discover the probabilistic equilibria, optimizing agents' action probabilities to augment the chances of a coordinated outcome, which in turn maximizes the team's total rewards. Given the presence of multiple sub-tasks and random key placements, this game can also be modeled as potential games, allowing agents to have varied rewards, missions, and objectives to coordinate.

> **Note**: When such games have multiple Nash equilibria, learning agents tend to converge to less risky equilibria by preferring less risky actions  - “Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks.” (2021)




### **Sub-Task 2.1** - Train your Baseline CTDE Agents

After successfully implementing `CentralizedCritic` in Task 1, take a look at the modified training command below:

```shell
python multigrid/scripts/train.py --local-mode False --env MultiGrid-CompetativeRedBlueDoor-v3-CTDE-Red --num-workers 10 --num-gpus 0 --name CTDE-Red_baseline --training-scheme CTDE  --training-config '{\"team_policies_mapping\": {\"red_0\" : \"your_policy_name\" , \"red_1\" : \"your_policy_name_v2\" }}' --restore-all-policies-from-checkpoint False 
```

### New Changes Walkthrough:

1. **`--training-config` (New✨)**: Optionally type `team_policies_mapping` as above or not specify. If not specify, `train.py` will refer to values in [training_config.json](submission/configs/training_config.json) directly.
2. **Customized Policies in `agents_pool` (New✨)**: `team_policies_mapping` in `training_config.json` links to customized policies registered in `SubmissionPolicies` in [multigrid/agents_pool/__init__.py](multigrid/agents_pool/__init__.py).
3. **Sample Policy Folder (New✨)**: Explore the [YourName_policies](multigrid/agents_pool/YourName_policies) folder in [multigrid/agents_pool](multigrid/agents_pool) for examples of customized "[Policy](multigrid/utils/policy.py)".
4. **Define Specific Values or Functions for Your Customized Policies**: In [YourPolicyName_Policy](multigrid/agents_pool/YourName_policies/YourPolicyName_policy.py) or [YourPolicyNameV2_Policy](multigrid/agents_pool/YourName_policies/YourPolicyName_policy_v2.py), you can set values or functions like `reward_schemes`, `algorithm_training_config`, and others.



> **Note**: Customization on  `reward_schemes`, `algorithm_training_config` should be safe to do. However, customization on `custom_observation_space`, `custom_observations`, `custom_handle_steps` are experinmental. But you're encouraged to adjust or introduce additional parameters, rewards or features as required. 

> **Note**: Add your policy module (e.g., `multigrid/agents_pool/STR_policies`) and register it by updating `SubmissionPolicies` in `multigrid/agents_pool/__init__.py`. Ensure it's part of your submission.


**Before You Start Training:**

#### Tips: Enhance Training with Manual Curriculum Learning
**Training Recommendation:** The default environment configuration for `MultiGrid-CompetativeRedBlueDoor-v3-CTDE-Red` in [multigrid/envs/__init__.py](multigrid/envs/__init__.py) has `"randomization"` set to True. Initially train agents with `randomization` set to False, allowing agents to master basic coordination skills for this Dec-POMDP without randomness.

Once you are comfortable with your agents configuration, execute the above command to train a Baseline CTDE Agents. Look out for the Ray Tune Commandline reporter status, now including individual agent rewards, e.g., `red_0_reward_mean` and `red_1_reward_mean`.

``` shell
Number of trials: 1/1 (1 RUNNING)
+----------------------------------------------------------------------------+----------+-----------------+--------+------------------+------+----------+---------------------+---------------------+----------------------+----------------------+--------------------+
| Trial name                                                                 | status   | loc             |   iter |   total time (s) |   ts |   reward |   red_0_reward_mean |   red_1_reward_mean |   episode_reward_max |   episode_reward_min |   episode_len_mean |
|----------------------------------------------------------------------------+----------+-----------------+--------+------------------+------+----------+---------------------+---------------------+----------------------+----------------------+--------------------|
| CentralizedCritic_MultiGrid-CompetativeRedBlueDoor-v3-CTDE-Red_d0b66_00000 | RUNNING  | 127.0.0.1:76927 |      1 |          141.034 | 4000 | 0.961428 |               0.273 |            0.688428 |              1.48478 |               0.4895 |               1280 |
+----------------------------------------------------------------------------+----------+-----------------+--------+------------------+------+----------+---------------------+---------------------+----------------------+----------------------+--------------------+

```

Here is the Tensorboard regex for this task:
`ray/tune/episode_len_mean|episodes_total|/learner_stats/vf_explained_var|ray/tune/policy_reward_mean/red_*|ray/tune/info/learner/red_0/learner_stats/entropy$|ray/tune/info/learner/red_1/learner_stats/entropy$`

Monitor Tensorboard to see how well your agents perform. Once your agents are able to meet the baseline thresholds, you can stop the current run, set `randomization` to True for  `MultiGrid-CompetativeRedBlueDoor-v3-CTDE-Red`, and then start another run with the following command:


```shell
python multigrid/scripts/train.py --local-mode False --env MultiGrid-CompetativeRedBlueDoor-v3-CTDE-Red --num-workers 10 --num-gpus 0 --name CTDE-Red_baseline --training-scheme CTDE  --training-config '{\"team_policies_mapping\": {\"red_0\" : \"your_policy_name\" , \"red_1\" : \"your_policy_name_v2\" }}' --restore-all-policies-from-checkpoint True --policies-to-load red_0 red_1 --load-dir <File Path to your last Checkpoint> --num-timesteps <Update this if you already reached 1M timesteps from your previous run. For example, extends it to 2e6 >
```

In this case, you will resume your training (In Blue) from your previous run (In Orange) but the key placement randomization will be activated in the training scenario.



<figure style="text-align: center;">
    <img src="images/HW_images/HW3/Task1_baseline.png" alt="Local GIF" style="width:2000px;"/>
    <figcaption style="text-align: center;">Tensorboard Plots for Manual Checkpoint Resume</figcaption>
</figure>


**Reflection Question**: What if I train CTDE agents with randomness from scratch?  

You can, but it will take a much longer time for the agents to start converging to some solutions. Take a look at this experinemnt (In Red) where we trained the CTDE agents from scratch.


<figure style="text-align: center;">
    <img src="images/HW_images/HW3/Task2_baseline.png" alt="Local GIF" style="width:2000px;"/>
    <figcaption style="text-align: center;">Tensorboard Plots for Manual Checkpoint Resume</figcaption>
</figure>



**CTDE Agent Training Baseline Thresholds:**
 
  | Metric                        | Expected Value   | Duration          |
  | ----------------------------- | ---------------- | ----------------- |
  | `episode_len_mean`            | 40 time steps    | Last 100k steps   |
  | `policy_reward_mean/red_*`    | 1.3+ returns     | Last 100k steps   |
  | `red_*/explained_variance`    | Above 0.9        | Last 100k steps   |
  | `red_*/learner_stats/entropy` | Below 0.4        | Last 100k steps   |

---






```shell

Number of trials: 1/1 (1 RUNNING)
+--------------------------------------------------------------+----------+-------------------+--------+------------------+------+-----------+------------------+-------------------+------------+---------------+----------------------+----------------------+--------------------+
| Trial name                                                   | status   | loc               |   iter |   total time (s) |   ts |    reward |   train_episodes |   red_reward_mean |   win_rate |   league_size |   episode_reward_max |   episode_reward_min |   episode_len_mean |
|--------------------------------------------------------------+----------+-------------------+--------+------------------+------+-----------+------------------+-------------------+------------+---------------+----------------------+----------------------+--------------------|
| PPO_MultiGrid-CompetativeRedBlueDoor-v3-CTCE-2v2_f8268_00000 | RUNNING  | 10.1.60.66:794634 |      1 |          127.513 | 4000 | -0.707352 |                4 |         0.0496484 |       0.75 |             3 |                 0.14 |             -1.73656 |              848.5 |
+--------------------------------------------------------------+----------+-------------------+--------+------------------+------+-----------+------------------+-------------------+------------+---------------+----------------------+----------------------+--------------------+


```



First, check out the CleanRL PPO implementation and its configuration in [`multigrid/scripts/train_ppo_cleanrl.py`](multigrid/scripts/train_ppo_cleanrl.py). You can do this by running the following command with the `--debug-mode True` flag.

Executing this command will display the default values of the training configuration and export a video showcasing the training scenario using random actions.

Command for Task 1:
```shell
python multigrid/scripts/train_ppo_cleanrl.py --debug-mode True 
```


### Task 1 Questions
After running the above command, observe the outputs in the command line. This will provide essential information required to train your RL agent.




#### Questions for General Deep RL Training Parameters Understanding
**Q.1** From the command line outputs, can you report the values for the following parameters from the command line outputs? Additionally, please describe the role of each parameter in the training loop and explain how these values influence training in a sentence or two. This exercise can help you grasp the fundamentals of `Sample Efficiency` and understand the tradeoffs when scaling your training process in a parallel fashion.  

- **num_envs**: 
- **batch_size**: 
- **num_minibatches**: 
- **minibatch_size**: 
- **total_timesteps**: 
- **num_updates**: 
- **num_steps**: 
- **update_epochs**: 


> **Note**: From Week 1, recall that `Sample Efficiency` refers to the ability of an algorithm to converge to an optimal solution with minimal sampling of experience data (trajectory from steps) from the environment.



#### Tips:
- Refer to [Part 1 of 3 — Proximal Policy Optimization Implementation: 11 Core Implementation Details](https://www.youtube.com/watch?v=MEt6rrxH8W4) from Week 2's Curriculum.
- Extensive comments and docstrings have been added atop the original [CleanRL ppo.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py) for your reference.
- Explore different configurations for the V2 environment in `CONFIGURATIONS` within [envs/__init__.py].
- Feel free to experiment with various arguments in [`multigrid/scripts/train_ppo_cleanrl.py`](multigrid/scripts/train_ppo_cleanrl.py) to familiarize yourself with this training script, its parameters, and the significance of the command line outputs.

#### Notes:
1. We only utilize the CleanRL PPO implementation in the first three main tasks of HW2. However, it offers a clean and straightforward way to grasp the ins and outs of the algorithm.
2. It's beneficial to explore other PPO implementations in CleanRL's official repository. For example:
    - [ppo_atari.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py)
    - [ppo_atari_lstm.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_lstm.py)
    - [ppo_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py)



---
## Task 2 - Understand the Deep RL Training Loop Dataflow & Implement Techniques to Minimize Learning Variance

In this task, you will delve into the specifics of the vectorized training architecture, which consists of two pivotal phases: the `Rollout Phase` and the `Learning Phase`. This is the parallelized training architecture that many Deep RL algorithms, including PPO used. You will also explore the techniques employed by PPO to reduce variance in learning, particularly focusing on the Generalized Advantage Estimation (GAE). You will enhance your understanding by identifying these phases in the code and implementing GAE to reduce variance of the training data before the `Learning Phase` when using the diversed data collected from the `Rollout Phase`.

### Questions to Enhance Understanding of the Deep RL Training Loop
***Q.1*** As mentioned in [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/), PPO employs a streamlined paradigm known as the vectorized architecture. This architecture encompasses two phases within the training loop:

- **Rollout Phase**: During this phase, the agent samples actions for 'N' environments and continues to process them for a designated 'M' number of steps.

- **Learning Phase**: In this phase, fundamentally, the agent learns from the data collected during the rollout phase. This data, with a length of NM, includes 'next_obs' and 'done'.

Utilizing your baseline codebase tagged `v2.1`, please pinpoint the `Rollout Phase` and the `Learning Phase` within the codebase, indicating specific line numbers. 

* For instance, the lines [189-211 in CleanRL ppo.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py#L189-L211) represent the Rollout Phase in their PPO implementation.  



**Q.2 How does PPO Reduce Variance? By Utilizing Generalized Advantage Estimation (GAE)? What is that?**

> **Note**: 
  PPO employs the Generalized Advantage Estimation (GAE) method for advantage calculation, merging multiple n-step estimators into a singular estimate, thereby mitigating variance and fostering more stable and efficient training.

  GAE amalgamates multiple n-step advantage estimators into a singular weighted estimator represented as:
  
      A_t^GAE(γ,λ) = Σ(γλ)^i δ_(t+i)

  
  where:
  
      δ_t - The temporal difference error formally defined as δ_t = r_t + γV(s_(t+1)) - V(s_t)
      γ - Discount factor which determines the weight of future rewards
      λ - A hyperparameter in [0,1] balancing bias and variance in the advantage estimation

  **References**:
  "High-Dimensional Continuous Control Using Generalized Advantage Estimation" by John Schulman et al.


If you run the following training command to train an agent, you are expected to see ValueErrors from blanks that needed to be filled to implement and enable Generalized Advantage Estimation (GAE). Please make use of the comments in the code to help you to implement GAE. 



Command for Task 2:
```shell
python multigrid/scripts/train_ppo_cleanrl.py --env-id MultiGrid-CompetativeRedBlueDoor-v2-DTDE-Red-Single-with-Obstacle --num-envs 8 --num-steps 128 --learning-rate 3e-4 --total-timesteps 10000000 --exp-name baseline
```

#### Tips:
- Useful comments has been added to the code for your guidance.
- For further insight, you might refer to the ["Generalized Advantage Estimation" section in "The 37 Implementation Details of Proximal Policy Optimization"](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/).


---

## Task 3 - Tuning the 🎲 **Exploration & Exploitation Strategies** using Algorithm-Specific Hyperparameters

Having implemented GAE in Task 2, re-run the training command provided below to start agent training. You're encouraged to adjust or introduce additional parameters as required.

```shell
python multigrid/scripts/train_ppo_cleanrl.py --env-id MultiGrid-CompetativeRedBlueDoor-v2-DTDE-Red-Single-with-Obstacle --num-envs 8 --num-steps 128 --learning-rate 3e-4 --total-timesteps 10000000 --exp-name baseline
```

### Deepening Your Understanding to Interpret Your Results
***Q.1*** Train a baseline agent using default or adjusted parameter values. Capture and present Tensorboard screenshots to report the following training metrics. Indicate the `Sample Effiicency`, the number of training timesteps and policy updates, required to achieve the Training Baseline Thresholds:

- **episodic_length**
- **episodic_return**
- **Policy_updates**
- **entropy**
- **explained_variance**
- **value_loss**
- **policy_loss**
- **approx_kl**

**CleanRL Agent Training Baseline Thresholds for Your Reference**:
- `episodic_length` should converge to a solution within 40 time steps and maintain for at least 100k time steps at the end of training.
- `episodic_return` should converge to consistently achieve 2.0+ returns, enduring for a minimum of the last 100k time steps.
- `explained_variance` should stabilize at a level above 0.6 for at least the last 100k time steps.
- `entropy` should settle at values below 0.3 for a minimum of 100k final training steps.

### Hands-on Experiences on PPO-Specific Hyperparameters
***Q.2*** If your baseline agent struggles to achieve the Training Baseline Thresholds, or if there's potential for enhancment, now you are getting the chance to fine-tuning the following PPO-specific parameters discussed in class to improve the performance of your agent. You may want to run multiple versions of experinements, so remember to modify `--exp-name` to differentiate between agent configurations. For final submissions, pick the top 3 performing or representable results and present the training metrics via screenshots and specify the number of timesteps and policy updates needed to fulfill or surpass the Training Baseline Thresholds. (Including links to their videos will be ideal)

- **gamma**
- **gae-lambda**
- **clip-coef**
- **clip-vloss**
- **ent-coef**
- **vf-coef**
- **target-kl**

Additionally, consider tweaking the following generic Deep RL hyperparameters:

- **num_envs**
- **batch_size**
- **num_minibatches**
- **minibatch_size**
- **total_timesteps**
- **num_updates**
- **num_steps**
- **update_epochs**


**Tips:**
- Monitor and track your runs using Tensorboard with the following command:
  ```shell
  tensorboard --logdir submission/cleanRL/runs
  ```
- You can filter the plots using the following filters:
  ```
  episodic_length|episodic_return|Policy_updates|entropy|explained_variance|value_loss|policy_loss|approx_kl
  ```
- Please refer the [Appendix](#appendix) for the definition of the PPO-specific parameters
- As mentioned in [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/): 
  - The significance of Value Function Loss Clipping is debatable. Engstrom, Ilyas, et al., (2020) didn't find evidence supporting its performance boost. In contrast, Andrychowicz, et al. (2021) inferred it might even hurt overall performance.
  - For Early Stopping, consider setting the target kl to `0.01`, as demonstrated in [OpenAI's PPO PyTorch Implementation](https://spinningup.openai.com/en/latest/algorithms/ppo.html#documentation-pytorch-version). 
  - `CompetativeRedBlueDoorEnvV2` is redundent on purpose so that you can modify it with the Deep RL knowledge you have learned so far to solve the `MultiGrid-CompetativeRedBlueDoor-v2-DTDE-Red-Single-with-Obstacle` scenario. 


---
## Extra Challenge #1: Mastering a Scenario with Sparse Learning Signals and Lower Complexity

Are you interested in challenging yourself further? Try to solve the scenario `MultiGrid-CompetitiveRedBlueDoor-v2-DTDE-Red-Single`. Intriguingly, the default agent configuration can solve the scenario with obstacle easily, but falters when there are none. 


See if you can solve the `MultiGrid-CompetativeRedBlueDoor-v2-DTDE-Red-Single`. Supprisingly, the default agent config can solve the scenario with obsticles, but not the scenario without. One possible explanation could be the close proximity of the obstacle to the door. Coupled with the annealing `ball_pickup_dense_reward`, these additional observations might act as additional learning signal, helping the agent overcome partial observability challenges and identify the sparse goal: opening the door using the key. But is this theory grounded in reality?

We invite you to investigate this by addressing the same questions outlined in Task 3. Share your findings and discoveries from this challenge in your `HW2_Answer.md` file.


**Tips**:
- Be on the lookout for potential bugs lurking within the training code or the environment. Andy Jones' insightful piece, [Debugging RL, Without the Agonizing Pain](https://andyljones.com/posts/rl-debugging.html), can be a valuable resource in debugging the environment.
- The `CompetitiveRedBlueDoorEnvV2` has been intentionally designed to be modifiable, and is not monitored by the CI/CD pipeline, providing you with the freedom to apply the deep RL concepts, you've learned so far to successfully navigate the `MultiGrid-CompetitiveRedBlueDoor-v2-DTDE-Red-Single` scenario.
- Similarly, the `CompetitiveRedBlueDoorWrapperV2` was created to allow for customization, enabling you to alter the raw observations received from the unwrapped upstream environment. This grants you the flexibility to define your very own observation and action spaces.

---

## Task 4: Bring the Lessons Learned from CleanRL to RLlib to solve a 1v1, 🤖 🆚 🤖 Scenario 

As you get familiar with PPO by working through the CleanRL implementation, let's pivot back to RLlib. We'll harness our understanding of hyperparameter tuning to address a 1v1 competition with a pre-trained opponent.

### 🎮 Visualizing the Scenario:

Kick-off by visualizing this new environment:

```shell
python multigrid/scripts/manual_control.py --env-id MultiGrid-CompetativeRedBlueDoor-v3-DTDE-1v1
```

In this death match scenario, your 'Red' agent will play against a pre-trained 'Blue' agent. The objective can be achieved in two ways:

  1. Grab the Red key and then unlock the Red door.
  2. Eliminating the Blue agent and use the [`toggle`](multigrid/core/actions.py) action, effectively trapping the Blue agent in their locked Blue room.

> **Note**: The `manual_control.py` script might crash when trying to navigate the scenario manually. This is due to the multi-agent nature of the `MultiGrid-CompetativeRedBlueDoor-v3-DTDE-1v1` scenario. Current support doesn't extend to controlling multiple agents with diverse actions.


### Starting Training:

Once you are familiar with the new scenario `MultiGrid-CompetativeRedBlueDoor-v3-DTDE-1v1`, run the following command to train a baseline agent:

```shell
python multigrid/scripts/train.py --local-mode False --env MultiGrid-CompetativeRedBlueDoor-v3-DTDE-1v1 --num-workers 10 --num-gpus 0 --name 1v1_death_match_baseline --training-scheme DTDE --policies-to-train red_0  --policies-to-load blue_0 --load-dir submission/pretrained_checkpoints/PPO_MultiGrid-CompetativeRedBlueDoor-v3-DTDE-1v1_154ab_00000_0_2023-09-12_16-08-06/checkpoint_000250
```

### Q.1 Metrics to Report:

As the same as Task 2&3, document the following training metrics, showcasing them with screenshots. Also, detail the number of timesteps and policy updates that meet or exceed the Training Baseline Thresholds.

Here are the RLLib and Scenario specific metrics:

- **episode_len_mean**
- **ray/tune/policy_reward_mean/red_0**
- **ray/tune/policy_reward_mean/blue_0**
- **ray/tune/sampler_results/custom_metrics/red_0/door_open_done_mean**
- **ray/tune/sampler_results/custom_metrics/blue_0/door_open_done_mean**
- **ray/tune/sampler_results/custom_metrics/red_0/eliminated_opponents_done_mean**
- **ray/tune/sampler_results/custom_metrics/blue_0/eliminated_opponents_done_mean**
- **ray/tune/counters/num_agent_steps_trained**
- **ray/tune/counters/num_agent_steps_sampled**
- **ray/tune/counters/num_env_steps_sampled**
- **ray/tune/counters/num_env_steps_trained**
- **episodes_total**
- **episodes_this_iter**
- **red_0/learner_stats/cur_kl_coeff**
- **red_0/learner_stats/entropy**
- **red_0/learner_stats/grad_gnorm**
- **red_0/learner_stats/kl**
- **red_0/learner_stats/policy_loss**
- **red_0/learner_stats/total_loss**
- **red_0/learner_stats/vf_explained_var**
- **red_0/learner_stats/vf_loss**
- **red_0/num_grad_updates_lifetime**



Here are the PPO-specific parameters in RLLib:
- **gamma**
- **lambda_**
- **kl_coeff**
- **kl_target**
- **clip_param**
- **grad_clip**
- **vf_clip_param**
- **vf_loss_coeff**
- **entropy_coeff**

> **Note**: [submission/configs/algorithm_training_config.json](submission/configs/algorithm_training_config.json) is where the training script calling the algorithm specific parameters from. Default values of PG and PPO specific parameters are stored in there as baselines for you.


**RLlib Agent Training Baseline Thresholds for Your Reference**:
- `episode_len_mean` should converge to a solution within 20 time steps and maintain for at least 100k time steps at the end of training.
- `ray/tune/policy_reward_mean/red_0` should converge to consistently achieve 1.3+ returns, enduring for a minimum of the last 100k time steps.
- `explained_variance` should stabilize at a level above 0.4 for at least the last 100k time steps.
- `red_0/learner_stats/entropy` should settle at values below 0.3 for a minimum of 100k final training steps.

**RLlib Agent Behavior Analysis Thresholds**
The following Metrics are Behavior-specific metrics. It depends on how your agent emerges into certain specific behaviors to achieve the RL objective to maximize the discounted sum of rewards from time step t to the end of the game. So, how to achieve the maximum return depends on the training environment's world dynamic and the agent's reward structures. So, the "Player Archetypes" of your agent can be varied. 

Our training scenario can be interpreted as a Zero-Sum game. Therefore, if your agent learned to solve a particular scenario by unlocking the door first, your Red agent should dominate this metric. Vice Versa.
- **ray/tune/sampler_results/custom_metrics/red_0/door_open_done_mean**
- **ray/tune/sampler_results/custom_metrics/blue_0/door_open_done_mean**

As mentioned above, if your agent learned to solve a particular scenario by eliminating the opponent first, your Red agent should dominate this metric. Vice Versa.
- **ray/tune/sampler_results/custom_metrics/red_0/eliminated_opponents_done_mean**
- **ray/tune/sampler_results/custom_metrics/blue_0/eliminated_opponents_done_mean**


For final submitions, pick the top 3 performing or representable results and present the training metrics via screenshots and specify the number of timesteps and policy updates needed to fulfill or surpass the Training Baseline Thresholds


**Tips:**
- Take a look at the configuration of `MultiGrid-CompetativeRedBlueDoor-v3-DTDE-1v1` in  [envs/__init__.py](multigrid/envs/__init__.py)
- You can filter the plots using the following filters:

```
eliminated_opponents_done_mean|episode_len_mean|num_agent_steps_trained|num_agent_steps_sampled|num_env_steps_sampled|num_env_steps_trained|episodes_total|red_0/learner_stats/cur_kl_coeff|red_0/learner_stats/entropy|red_0/learner_stats/grad_gnorm|red_0/learner_stats/kl|red_0/learner_stats/policy_loss|red_0/learner_stats/total_loss|red_0/learner_stats/vf_explained_var|red_0/learner_stats/vf_loss|red_0/num_grad_updates_lifetime|ray/tune/policy_reward_mean/red_0|ray/tune/policy_reward_mean/blue_0
```
- RLlib Tune may report metrics with different names but pointing to the same metric. For example, `ray/tune/sampler_results/custom_metrics/blue_0/door_open_done_mean` is the same as `ray/tune/custom_metrics/blue_0/door_open_done_mean` so just report one is fine.


- To visualize a specific checkpoint, use the following command:
```shell
python multigrid/scripts/visualize.py --env MultiGrid-CompetativeRedBlueDoor-v3-DTDE-1v1  --num-episodes 10  --load-dir submission/ray_results/PPO/PPO_MultiGrid-CompetativeRedBlueDoor-v3-DTDE-1v1_XXXX/checkpoint_YYY/checkpoint-YYY --render-mode human --gif DTDE-1v1-testing
```
##### Replace `XXXX` and `YYY` with the corresponding number of your checkpoint.


- If running on Colab, use the `%tensorboard` [line magic](https://ipython.readthedocs.io/en/stable/interactive/magics.html) to achieve the same; see the [notebook](notebooks/homework1.ipynb) for more details.



---
## Extra Challenge #2: Exploring Different RLLib Algorithms
Are you interested in challenging yourself further? See if you can solve the same scenario using alternative [Deep RL Algorithms](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#available-algorithms-overview) provided by RLLib.

To experiment with different algorithms, adjust the `--algo` flag to specify an alternate RLlib-registered algorithm.

**Tips**:
- Review the supported conditions for each algorithm. Look for specific requirements or features like `Discrete Actions`, `Continuous Actions`, `Multi-Agent`, and `Model Support`.
- Be aware: this RL class codebase is tested for PPO. Switching directly to other algorithms may introduce unexpected issues. The most straightforward adjustments might involve changing algorithm-specific parameters. More complex modifications could entail code changes to accommodate specific observation or action spaces.



---

## Task 5 - Homework Submission via Github Classroom

### Submission Requirements:

1. **CleanRL Agent**: 
    - Commit and push your best-performing cleanRL agent, ensuring it meets the minimum required thresholds described in the Task, to [submission/cleanRL](submission/cleanRL).
    - For videos, save them to [submission/cleanRL/videos](submission/cleanRL/videos). Please be mindful regarding video size and retain only the most representative ones. Rename the videos as needed for clarity.

2. **RLlib Agents**: 
    - Commit and push your best-performing RLlib agents and checkpoints, ensuring they satisfy the minimum thresholds described in the Task, to [submission/ray_results](submission/ray_results). And also your customized [submission/configs](submission/configs).

3. **RLlib Agents Evaluation Reports**: 
    - Commit and push relevant RLlib agent evaluation results: `<my_experiment>_eval_summary.csv`, `<my_experiment>_episodes_data.csv`, and `<my_experiment>.gif` to [submission/evaluation_reports](submission/evaluation_reports).

4. **Answers to Questions**:
    - For question answers, either:
      - Update the provided [homework2.ipynb](notebooks/homework2.ipynb) notebook, or 
      - Submit a separate `HW2_Answer.md` file under [submission](submission).

5. **MLFlow Artifacts**:
    - Ensure you commit and push the MLFlow artifacts to [submission](submission) (Which should be automatic).


#### Tips:
- Retain only the top-performing checkpoints in [submission/ray_results](submission/ray_results).
    - Refer to the baseline performance thresholds specified for each agent training task.
    - Uploading numerous checkpoints, particularly underperforming ones, may cause the CI/CD to fail silently due to time constraints.
    
- Executing [tests/test_evaluation.py](tests/test_evaluation.py) with `pytest` should generate and push the necessary results to [submission/evaluation_reports](submission/evaluation_reports).

- For an exemplar submission that fulfills all the requirements and successfully passing the Autograding Github Actions, please checkout [Example Submission](https://github.com/STRDeepRL/week-1-intro-to-deep-rl-and-agent-training-environments-heng4str).

- Always place your submissions within the `submission/` directory. If opting for the notebook approach, please maintain your edited `homework2.ipynb` and related documents under `notebooks/`.

- **Honesty System**: If OS compatibility issues hinder task completion, you're permitted to modify files outside the `EXCEPTION_FILES` listed in [tests/test_codebase.py](tests/test_codebase.py). Add those modified files to the list in your own `test_codebase.py`. However, ensure these changes don't impact your Agent Training Performance, as the centralized evaluation in Week 4's Agent Competition won't consider these changes.

- If you would like to showcase your work at the begining of the class, please notify the class facilitators in advance.


***Note:*** 
Please beaware that the [File Size Check GitHub Action Workflow](.github/workflows/check_file_size.yml) will check the total files size for folers "submission/" "notebooks/", to ensure each of them will not exceed 5MBs. Please ensure to only submit the checkpoints, the notebooks and the MLFlow artifacts that are meant for grading by the Github Action CI/CD pipeline.



---

## Appendix:

Here are definition of the Training Metrics in CleanRL:
- **episodic_length**: 
  - **Definition**: The number of time steps taken by the agent in an episode before it terminates (either by achieving its goal, failing, or reaching a maximum time step limit).
  - **Relevance**: A shorter episodic length, especially in tasks where the goal is to achieve something in minimal time, can indicate more efficient learning and problem solving by the agent.

- **episodic_return**: 
  - **Definition**: The cumulative reward obtained by the agent over a complete episode. It's a discounted sum of rewards received at each time step during an episode.
  - **Relevance**: Higher episodic returns indicate better policy performance, with the agent effectively maximizing its cumulative reward.

- **Policy_updates**: 
  - **Definition**: The number of times the agent's policy has been updated during training. This is usually incremented each time backpropagation is performed to update the policy's parameters.
  - **Relevance**: Keeping track of policy updates can help in analyzing the agent's convergence rate and the effect of each update on overall performance.

- **entropy**: 
  - **Definition**: A measure of the randomness in the agent's policy. In the context of RL, it quantifies the uncertainty in an agent's actions.
  - **Relevance**: Encouraging higher entropy can help promote exploration, while lower entropy indicates a more deterministic policy. Balancing exploration (high entropy) and exploitation (low entropy) is crucial in many tasks.

- **explained_variance**: 
  - **Definition**: A statistical measure that captures the proportion of the variance in the dependent variable that's "explained" by the independent variables in a regression model.
  - **Relevance**: In RL, it can indicate how well the value function approximates the expected return. A value closer to 1 suggests the value function is a good predictor, whereas values closer to 0 suggest poor prediction.

- **value_loss**: 
  - **Definition**: The discrepancy between the predicted value of states (by the value function) and the actual returns observed. Typically calculated as the mean squared error between these two.
  - **Relevance**: Minimizing value loss ensures the agent has an accurate estimate of future rewards, which is crucial for making optimal decisions.

- **policy_loss**: 
  - **Definition**: Represents the loss in the agent's policy. It's often calculated based on how much the current policy deviates from a previously successful policy (in PPO, this is related to the clipped objective function).
  - **Relevance**: Monitoring policy loss ensures that updates to the policy push it towards better performance without drastic changes that might destabilize learning.

- **approx_kl**: 
  - **Definition**: Short for "approximate Kullback-Leibler divergence." It measures how one probability distribution diverges from a second, expected probability distribution. In PPO, it's used to quantify the difference between the old and new policies.
  - **Relevance**: Keeping the KL divergence low ensures that the policy doesn't change too dramatically during updates, preserving stability in training.


Here are definition of the PPO-specific parameters used in CleanRL:

- **gamma** (`γ`):
  - **Definition**: Known as the discount factor, it's a number between 0 and 1 that represents the agent's consideration for future rewards. 
  - **Relevance**: A higher `γ` makes the agent prioritize long-term reward over short-term reward, while a lower value does the opposite. Tuning `γ` affects how the agent balances immediate vs. future rewards.

- **gae-lambda** (`λ`):
  - **Definition**: Used in Generalized Advantage Estimation (GAE). It's a factor in the range of 0 and 1 that determines the trade-off between using more of the raw rewards (`λ = 0`) versus more of the estimated value function (`λ = 1`) when computing the advantage.
  - **Relevance**: Adjusting `λ` can help strike a balance between bias and variance in the advantage estimate, potentially stabilizing training and improving performance.

- **clip-coef**:
  - **Definition**: In PPO, it's the epsilon in the objective function's clipping mechanism. The objective is clipped to be within the range of `(1-epsilon, 1+epsilon)`, preventing large policy updates.
  - **Relevance**: This coefficient prevents overly aggressive updates to the policy, ensuring stable training. Changing its value affects the size of acceptable policy updates.

- **clip-vloss**:
  - **Definition**: Value function loss clipping coefficient. It restricts the value function's update to keep the loss changes within a certain range.
  - **Relevance**: Similar to the policy clipping mechanism, this coefficient can be adjusted to stabilize the training of the value function, especially when significant value changes are observed.

- **ent-coef**:
  - **Definition**: Entropy coefficient. It scales the entropy bonus in the PPO objective function.
  - **Relevance**: Adjusting this coefficient affects the balance between exploration and exploitation. A higher entropy coefficient encourages more exploration.

- **vf-coef**:
  - **Definition**: Value function coefficient. It's a scaling factor that determines the weight of the value function loss in the overall PPO loss.
  - **Relevance**: Adjusting this parameter can balance the importance of the value function loss compared to the policy loss, affecting how the agent learns to estimate state values versus improving its policy.

- **target-kl**:
  - **Definition**: Target Kullback-Leibler (KL) divergence. It's a threshold used in some PPO implementations to apply early stopping to prevent large policy updates. If the KL divergence between the new and old policy exceeds this target, the update can be skipped or the learning rate can be adjusted.
  - **Relevance**: Ensures policy updates do not stray too far from the original policy, maintaining training stability.



Here are definition of the PPO-specific parameters used in RLLib:

- **gamma** (`γ`):
  - **Definition**: Known as the discount factor, it's a number between 0 and 1 that represents the agent's consideration for future rewards. 
  - **Relevance**: A higher `γ` makes the agent prioritize long-term reward over short-term reward, while a lower value does the opposite. Tuning `γ` affects how the agent balances immediate vs. future rewards.

- **lambda_**:
  - **Definition**: Used in Generalized Advantage Estimation (GAE). It's a factor in the range of 0 and 1 that determines the trade-off between using more of the raw rewards (`λ = 0`) versus more of the estimated value function (`λ = 1`) when computing the advantage.
  - **Relevance**: Adjusting `λ` can help strike a balance between bias and variance in the advantage estimate, potentially stabilizing training and improving performance.

- **kl_coeff**:
  - **Definition**: A scaling factor on the KL-divergence term in the objective. KL-divergence measures the difference between the new and old policy distributions.
  - **Relevance**: Balances the KL penalty with the PPO objective. Adjusting this influences the magnitude of policy updates, potentially affecting training stability.

- **kl_target**:
  - **Definition**: The desired KL divergence between the old and new policy. 
  - **Relevance**: Acts as a regulator for `kl_coeff`. If KL divergence drifts from this target, `kl_coeff` is adjusted to bring it back, ensuring policy updates remain controlled.

- **clip_param**:
  - **Definition**: The epsilon value for PPO's clipping mechanism. It bounds the ratio of policy probabilities to ensure limited policy updates.
  - **Relevance**: It prevents excessively large policy updates, ensuring stability during training.

- **grad_clip**:
  - **Definition**: Parameter that determines the maximum allowed gradient norm during training.
  - **Relevance**: Clipping the gradients prevents large updates, offering more stable training, especially in scenarios with sharp loss landscapes.

- **vf_clip_param**:
  - **Definition**: The epsilon value for clipping the value function updates.
  - **Relevance**: It restricts the magnitude of value function updates, adding stability to the learning process.

- **vf_loss_coeff**:
  - **Definition**: Coefficient for the value function loss in the PPO loss function.
  - **Relevance**: Balances the importance of value function updates compared to policy updates, influencing how the agent trades off between value estimation and policy improvement.

- **entropy_coeff**:
  - **Definition**: Coefficient to scale the entropy bonus term in the PPO objective.
  - **Relevance**: Entropy encourages exploration, so adjusting this parameter can influence how much the agent explores the environment versus exploiting known strategies.