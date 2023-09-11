from __future__ import annotations

""" Expected for restricted changes """


"""Script for Training Deep Reinforcement Learning agents in MultiGrid environment wiht Proximal Policy Optimization.

Documentation and experiment results related to this implementation can be found at: https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy

Note: This script is expected to have restricted changes.

"""

# Imports 
import argparse
import subprocess
import os
import random
import time
from distutils.util import strtobool
import pprint

pp = pprint.PrettyPrinter(indent=4)

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from multigrid.envs import *
from multigrid.wrappers import SingleAgentWrapperV2, CompetativeRedBlueDoorWrapperV2

# ======== Define Global Variables and Configurations ======== #

# Find the root directory of the git repository and change the current working directory to that path.
REPO_ROOT = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).strip().decode("utf-8")
os.chdir(REPO_ROOT)

# Define the submission folder's path
SUBMISSION_FOLDER = "submission/cleanRL"

# Define the frequency to save checkpoints of the model during training.
CHECKPOINT_FREQUENCY = 50


def parse_args():
    """
    Parses command-line arguments for the PPO training script.
    
    Returns:
        argparse.Namespace: A namespace containing all the parsed arguments.
    """

    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--save-checkpoint", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, the checkpoint for this experiment will be tracked and save in checkpoints/")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="MultiGrid-CompetativeRedBlueDoor-v2-DTDE-Red-Single-with-Obsticle",  #"MultiGrid-CompetativeRedBlueDoor-v2-DTDE-Red-Single"
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=5000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=8,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.9,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=1.0,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01, #0.01
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=0.015, # None
        help="the target KL divergence threshold")
    parser.add_argument(
        "--debug-mode", type=bool, default=True, help="Boolean value to set to use debug mode for debugging"
    )
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    """
    Factory function to create and configure a gym environment.

    Parameters
    ----------
    env_id : str
        Identifier of the gym environment to be created.
    seed : int
        Seed for initializing the environment's random processes.
    idx : int
        Index of the environment instance, mainly used to determine if videos should be recorded.
    capture_video : bool
        Flag indicating whether to capture videos of the environment's episodes.
    run_name : str
        Name of the current run, used in naming video files.

    Returns
    -------
    func
        A function that, when called, initializes and returns the specified gym environment.

    Examples
    --------
    >>> env_factory = make_env('CartPole-v1', 0, 0, True, 'test_run')
    """
    
    def thunk():
        env = gym.make(
            env_id, agents=1, render_mode="rgb_array", screen_size=640, disable_env_checker=True
        )  #  render_mode="human",
        env = CompetativeRedBlueDoorWrapperV2(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"{SUBMISSION_FOLDER}/videos/{run_name}")
        env = SingleAgentWrapperV2(env)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Initializes a given neural network layer's weights and biases using orthogonal initialization.

    Parameters
    ----------
    layer : torch.nn.Module
        The neural network layer to be initialized.
    std : float, optional
        The standard deviation for the orthogonal initialization of weights. Default is sqrt(2).
    bias_const : float, optional
        The constant value for initializing biases. Default is 0.0.

    Returns
    -------
    torch.nn.Module
        The initialized neural network layer.

    Examples
    --------
    >>> linear_layer = torch.nn.Linear(10, 20)
    >>> initialized_layer = layer_init(linear_layer)

    Notes
    -----
    - Reference: Exact solutions to the nonlinear dynamics of learning in deep linear neural 
      networks - Saxe, A. et al. (2013).
    - The input tensor for the orthogonal initialization should have at least 2 dimensions. 
      For tensors with more than 2 dimensions, the trailing dimensions are flattened.

    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """
    Defines an agent using an Actor-Critic architecture to interact with an environment.

    The agent comprises two main components:
    - Actor: Outputs the action probabilities for a given state/observation.
    - Critic: Estimates the value of a given state/observation.

    Parameters
    ----------
    envs : gym.Env
        The environments with which the agent interacts.

    Attributes
    ----------
    critic : torch.nn.Sequential
        The neural network that estimates state/observation values.
    actor : torch.nn.Sequential
        The neural network that outputs action probabilities.

    Methods
    -------
    get_value(x: torch.Tensor) -> torch.Tensor:
        Returns the estimated value of the input state/observation.
    
    get_action_and_value(x: torch.Tensor, action: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        Returns the chosen action, its log probability, the entropy of the policy, and the state value.

    """
    def __init__(self, envs):
        super().__init__()
        # Define the critic network to estimate state/observation values
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        # Define the actor network to output action probabilities
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        """
        Estimates the value of the given state/observation.

        Parameters
        ----------
        x : torch.Tensor
            The state/observation tensor.

        Returns
        -------
        torch.Tensor
            The estimated value of the state/observation.
        """
        # Reshape the tensor to (batch_size, -1)
        x = x.view(x.size(0), -1)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """
        Computes the chosen action, its log probability, the entropy of the policy, and the state/observation value.

        Parameters
        ----------
        x : torch.Tensor
            The state/observation tensor.
        action : torch.Tensor, optional
            If provided, this action will be used. Otherwise, an action is sampled from the policy. 

        Returns
        -------
        torch.Tensor
            The chosen action.
        torch.Tensor
            The log probability of the chosen action.
        torch.Tensor
            The entropy of the policy.
        torch.Tensor
            The estimated value of the state/observation.
        """
        # Reshape the tensor to (batch_size, -1)
        x = x.view(x.size(0), -1)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


if __name__ == "__main__":
    args = parse_args()

    print("\n======== Algorithm Configurations ========")
    pp.pprint(vars(args))
    print("==========================================\n")

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.save_checkpoint:
        checkpoint_folder = f"{SUBMISSION_FOLDER}/checkpoints/{run_name}"

        checkpoint_folder = os.path.abspath(checkpoint_folder)
        # Create output folder if needed
        os.makedirs(checkpoint_folder, exist_ok=True)

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"{SUBMISSION_FOLDER}/runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    # Seeding for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Initialize vectorized environments
    """
    NOTE:
    The code uses a vectorized environment (SyncVectorEnv) from gym, which allows running multiple instances of an environment in parallel.
    This is particularly useful for algorithm like Proximal Policy Optimization (PPO), which can benefit from parallel environment interactions.
    The environments are created using the provided make_env function
    """
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)
        ],  
    )

    # Check and print environment details
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    print(f"Environment Details:")
    print(f" - Action space: {envs.single_action_space} with {envs.single_action_space.n} discrete actions")
    print(f" - Observation space: {envs.single_observation_space} with shape {envs.single_observation_space.shape}\n")

    # Initialize agent and print its architecture
    agent = Agent(envs).to(device)
    print("Agent Model Architecture:")
    print(agent)
    print("\n")
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    # Setup storage for agent's learning loop
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    # Initialize the game and fetch initial observations
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    # Print training initialization details
    print(f"Agent Training Initialization:")
    print(f" - Number of updates in this training cycle: {num_updates}")
    print(f" - Estimated State Value for the initial observation:\n")
    pp.pprint(agent.get_value(next_obs))
    print(f" - Shape of the estimated state value: {agent.get_value(next_obs).shape}\n")
    print(f" - Action and value details for the initial observation:\n")
    pp.pprint(agent.get_action_and_value(next_obs))
    print("\n")

    if args.debug_mode:
        # Running the envinorment in debug mode
        print("Running the environment...")
        next_obs = envs.reset()
        for _ in range(1000):
            action = envs.action_space.sample()
            next_obs, reward, done, truncate, info = envs.step(action)
            for item in info:
                if "final_info" in info:
                    episodic_returns = []
                    episodic_lengths = []
                    for sub_item in info["final_info"]:
                        if (sub_item is not None) and ("episode" in sub_item.keys()):
                            print(
                                f"Episode details: Return = {sub_item['episode']['r']}, Length = {sub_item['episode']['l']}"
                            )

    else:
        # Main Training Loop for updating the PPO policy
        """
        NOTE:
        This loop represents the number of updates to be made to the policy during training. 
        Each update consists of collecting some rollout trajectory data and then performing a series of optimizations on the collected data.

        Recalls:
        num_updates - The total number of times the policy (or agent) is updated throughout the entire training process
        num_steps - The number of consecutive interactions (steps) an agent takes in the environment before the data is used for an update
        update_epochs - 
        batch_size - 
        minibatch_size -
        """
        for update in range(1, num_updates + 1):
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                # Calculates the fraction of learning rate left
                # Update the learning rate with annealing
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            # Main Rollout loop for training trajectory data collection
            """
            NOTE:
            This loop is for collecting data from the environments using the current policy. 
            Data like observations, rewards, and actions are stored for later use during optimization
            """
            for step in range(0, args.num_steps):
                global_step += 1 * args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                # NOTE: Using the current policcy to get the action, value, and log probabilities of the next observation without calculating gradients
                #       It's just for data collection purpose
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                # NOTE: This block steps the environment with the selected action using the current policy
                next_obs, reward, done, truncate, info = envs.step(action.cpu().numpy())
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

                # Log episodic returns and lengths.
                if "final_info" in info:
                    episodic_returns = []
                    episodic_lengths = []
                    for item in info["final_info"]:
                        if (item is not None) and ("episode" in item.keys()):
                            writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                            writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                            break

            # Estimate the advantage using Generalized Advantage Estimation (GAE)
            """
            NOTE:
            This section calculates the advantages and returns which are essential parts of the PPO algorithm. 
            It uses the Generalized Advantage Estimation (GAE) method to calculate the advantages
            This block also bootstraps value if not done
            """
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # Flatten the batch to fit the neural network's input dimensions
            # NOTE: This part is reshaping the tensor structure for ease of processing
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # PPO optimization: Optimize the policy and value function
            # NOTE: This loop represents the number of epochs of optimization to perform on the collected data
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    # Obtain updated action and value predictions for the minibatch
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions.long()[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    # Gradient Descent
                    """ 
                    NOTE: 
                    Here the gradients are calculated and a step of optimization is performed. 
                    Gradients are clipped to prevent too large updates which can destabilize training.
                    """
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                # Early Stopping
                """
                NOTE: 
                If the average KL divergence between the new and old policy is larger than a threshold, it stops the optimization early. 
                This is to prevent too large policy updates which can destabilize the training
                """
                if args.target_kl is not None:
                    if approx_kl > args.target_kl:
                        break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # Logging Training Information
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            print("Steps per Second (SPS):", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            if args.save_checkpoint:
                # make sure to tune `CHECKPOINT_FREQUENCY`
                # so models are not saved too frequently
                if update % CHECKPOINT_FREQUENCY == 0:
                    torch.save(agent.state_dict(), f"{checkpoint_folder}/agent.pt")

    envs.close()
    writer.close()