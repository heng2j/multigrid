from __future__ import annotations

import argparse
import json
import os
import ray

from multigrid.utils.training_utilis import algorithm_config, can_use_gpu,  get_checkpoint_dir
from multigrid.rllib.models import TFModel, TorchModel, TorchLSTMModel
from pathlib import Path
from pprint import pprint
from ray import tune
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.from_config import NotProvided
from ray.tune.registry import get_trainable_cls
from ray.tune import CLIReporter
from ray.rllib.policy.policy import Policy
import ray.rllib.algorithms.callbacks as callbacks
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.callback import Callback
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib import BaseEnv, Policy, RolloutWorker
from typing import Dict, Optional, Union
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import AgentID, EnvType, PolicyID
import numpy as np
import json

import pathlib
SCRIPT_PATH = str(pathlib.Path(__file__).parent.absolute().parent.absolute())

import git

# Limit the number of rows.
reporter = CLIReporter(max_progress_rows=10, max_report_frequency=30)






tags = { "user_name" : "John",
         "git_commit_hash" : git.Repo(SCRIPT_PATH).head.commit
         }





# TODO - Set Evaluation
class EvaluationCallbacks(DefaultCallbacks, Callback):
    def on_episode_step(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        episode: Union[Episode, EpisodeV2],
        env_index: Optional[int] = None,
        **kwargs,
    ):
        info = episode._last_infos
        for a_key in info.keys():
            if a_key != "__common__":
                for b_key in info[a_key]:
                        try:
                            episode.user_data[f"{a_key}/{b_key}"].append(info[a_key][b_key])
                        except KeyError:
                            episode.user_data[f"{a_key}/{b_key}"] = [info[a_key][b_key]]

    def on_episode_end(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2, Exception],
        env_index: Optional[int] = None,
        **kwargs,
    ):
        info = episode._last_infos
        for a_key in info.keys():
            if a_key != "__common__":
                for b_key in info[a_key]:
                    metric = np.array(episode.user_data[f"{a_key}/{b_key}"])
                    episode.custom_metrics[f"{a_key}/{b_key}"] = np.sum(metric).item()


class RestoreWeightsCallback(DefaultCallbacks, Callback):

    def __init__(
            self,
            load_dir: str,
            policy_name: str,
        ):
        self.load_dir = load_dir
        self.policy_name = policy_name


    def on_algorithm_init(self, *, algorithm: "Algorithm", **kwargs) -> None:
        algorithm.set_weights({self.policy_name: self.restored_policy_0_weights})


    def setup(self, *args, **kwargs):
        policy_0_checkpoint_path = get_checkpoint_dir(self.load_dir)
        restored_policy_0 = Policy.from_checkpoint(policy_0_checkpoint_path)
        self.restored_policy_0_weights = restored_policy_0[self.policy_name].get_weights()




# TODO - Set CTDE Model
# https://github.com/ray-project/ray/blob/master/rllib/examples/centralized_critic.py
# https://github.com/ray-project/ray/blob/master/rllib/examples/models/centralized_critic_models.py


import numpy as np
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.torch_utils import convert_to_torch_tensor

from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing
from ray.rllib.algorithms.ppo.ppo import PPO, PPOConfig


GLOBAL_TEAM_OBS = "global_team_obs"
GLOBAL_TEAM_ACTION = "global_team_action"



class CentralizedValueMixin:
    """Add method to evaluate the central value function from the model."""

    def __init__(self):
        self.compute_central_vf = self.model.central_value_function



# Grabs the global team obs/act and includes it in the experience train_batch,
# and computes GAE using the central vf predictions.
def centralized_critic_postprocessing(
    policy, sample_batch, other_agent_batches=None, episode=None
):

    # Prep for our agent
    policy_id = policy._Policy__policy_id
    team_name = policy_id.split("_")[0]
    team_num = policy.config["env_config"]["teams"][team_name]
    team_members_agents = [ f"{team_name}_{i}" for i in range(team_num) if f"{team_name}_{i}"  != policy_id] 



    pytorch = policy.config["framework"] == "torch"
    if (pytorch and hasattr(policy, "compute_central_vf")) or (
        not pytorch and policy.loss_initialized()
    ):
        assert other_agent_batches is not None

        # also record the global team obs and actions in the trajectory
        sample_batch[GLOBAL_TEAM_OBS] = np.concatenate(
            [other_agent_batches[agent_id][2][SampleBatch.CUR_OBS] for agent_id in team_members_agents],
            axis=1)

        sample_batch[GLOBAL_TEAM_ACTION] = np.stack(
            [other_agent_batches[agent_id][2][SampleBatch.ACTIONS] for agent_id in team_members_agents],
            axis=1)

        # overwrite default VF prediction with the central VF
        sample_batch[SampleBatch.VF_PREDS] = (
            policy.compute_central_vf(
                convert_to_torch_tensor(
                    sample_batch[SampleBatch.CUR_OBS], policy.device
                ),
                convert_to_torch_tensor(sample_batch[GLOBAL_TEAM_OBS], policy.device),
                convert_to_torch_tensor(
                    sample_batch[GLOBAL_TEAM_ACTION], policy.device
                ),
            )
            .cpu()
            .detach()
            .numpy()
        )


    else:
        # Policy hasn't been initialized yet, use zeros.
        sample_batch[GLOBAL_TEAM_OBS] =  np.zeros((sample_batch[SampleBatch.CUR_OBS].shape[0], sample_batch[SampleBatch.CUR_OBS].shape[1]*(team_num -1))) #np.zeros_like(sample_batch[SampleBatch.CUR_OBS])
        sample_batch[GLOBAL_TEAM_ACTION] = np.zeros((sample_batch[SampleBatch.ACTIONS].shape[0], team_num - 1),) # np.zeros_like(sample_batch[SampleBatch.ACTIONS]) #  np.zeros((sample_batch[SampleBatch.ACTIONS].shape[0], sample_batch[SampleBatch.ACTIONS].shape[1] * (team_num-1))) #np.zeros_like(sample_batch[SampleBatch.ACTIONS])
        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(
            sample_batch[SampleBatch.REWARDS], dtype=np.float32
        )

    completed = sample_batch[SampleBatch.TERMINATEDS][-1]
    if completed:
        last_r = 0.0
    else:
        last_r = sample_batch[SampleBatch.VF_PREDS][-1]

    train_batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"],
    )
    return train_batch


# Copied from PPO but optimizing the central value function.
def loss_with_central_critic(policy, base_policy, model, dist_class, train_batch):
    # Save original value function.
    vf_saved = model.value_function

    # Calculate loss with a custom value function.
    model.value_function = lambda: policy.model.central_value_function(
        train_batch[SampleBatch.CUR_OBS],
        train_batch[GLOBAL_TEAM_OBS],
        train_batch[GLOBAL_TEAM_ACTION],
    )
    policy._central_value_out = model.value_function()
    loss = base_policy.loss(model, dist_class, train_batch)

    # Restore original value function.
    model.value_function = vf_saved

    return loss


class CTDEPPOTorchPolicy(CentralizedValueMixin, PPOTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        PPOTorchPolicy.__init__(self, observation_space, action_space, config)
        CentralizedValueMixin.__init__(self)

    @override(PPOTorchPolicy)
    def loss(self, model, dist_class, train_batch):
        return loss_with_central_critic(self, super(), model, dist_class, train_batch)

    @override(PPOTorchPolicy)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        return centralized_critic_postprocessing(
            self, sample_batch, other_agent_batches, episode
        )


class CentralizedCritic(PPO):
    @classmethod
    @override(PPO)
    def get_default_policy_class(cls, config):
        return CTDEPPOTorchPolicy




def train(
    algo: str,
    config: AlgorithmConfig,
    stop_conditions: dict,
    save_dir: str,
    load_dir: str | None = None,
    local_mode: bool = False,
    experiment_name: str = "testing_experiment",
    training_scheme: str = "CTDE", # Can be either "CTCE", "DTDE" or "CTDE"  
    ):
    """
    Train an RLlib algorithm.
    """



    ray.init(num_cpus=(config.num_rollout_workers + 1), local_mode=local_mode)
    tune.run(
        CentralizedCritic if training_scheme == "CTDE" else algo,
        stop=stop_conditions,
        config=config,
        local_dir=save_dir,
        verbose=1,
        restore=get_checkpoint_dir(load_dir),
        checkpoint_freq=10,
        checkpoint_at_end=True,
        progress_reporter=reporter,
        callbacks=[MLflowLoggerCallback(
        tracking_uri="./submission/mlflow",
        experiment_name=experiment_name,
            tags=tags,
            save_artifact=True),
          ] #   RestoreWeightsCallback(load_dir=load_dir,policy_name="policy_0"),
    )
    ray.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--algo', type=str, default='PPO',
        help="The name of the RLlib-registered algorithm to use.")
    parser.add_argument(
        '--framework', type=str, choices=['torch', 'tf', 'tf2'], default='torch',
        help="Deep learning framework to use.")
    parser.add_argument(
        '--lstm', action='store_true', help="Use LSTM model.")
    parser.add_argument(
        '--env', type=str, default='MultiGrid-CompetativeRedBlueDoor-v3-CTDE-Red',
        help="MultiGrid environment to use.")
    parser.add_argument(
        '--env-config', type=json.loads, default={},
        help="Environment config dict, given as a JSON string (e.g. '{\"size\": 8}')")
    parser.add_argument(
        '--num-agents', type=int, default=2, help="Number of agents in environment.") # FIXME - streamline this with teams
    parser.add_argument(
        '--seed', type=int, default=0, help="Set the random seed of each worker. This makes experiments reproducible")
    parser.add_argument(
        '--num-workers', type=int, default=40, help="Number of rollout workers.") 
    parser.add_argument(
        '--num-gpus', type=int, default=1, help="Number of GPUs to train on.")
    parser.add_argument(
        '--num-timesteps', type=int, default=1e6,
        help="Total number of timesteps to train.")
    parser.add_argument(
        '--lr', type=float, help="Learning rate for training.")
    parser.add_argument(
        '--load-dir', type=str, #default='/Users/zla0368/Documents/RL/RL_Class/code/multigrid/submission/ray_results/PPO/PPO_MultiGrid-CompetativeRedBlueDoor-v0_5dfa1_00000_0_2023-08-01_23-12-09',
        help="Checkpoint directory for loading pre-trained policies.")
    parser.add_argument(
        '--save-dir', type=str, default='submission/ray_results/',
        help="Directory for saving checkpoints, results, and trained policies.")
    parser.add_argument(
        '--name', type=str, default='<my_experinemnt>',
        help="Distinct name to track your experinemnt in save-dir")
    parser.add_argument(
        '--local-mode', type=bool, default=False,
        help="Boolean value to set to use local mode for debugging")
    parser.add_argument(
        '--our-agent-ids', nargs="+", type=int, default=[0,1],
        help="List of agent ids to train")
    parser.add_argument(
        '--teams', type=json.loads, default={"red": 2}, # "blue": 2
        help='A dictionary containing team name and counts, e.g. \'{"red": 2, "blue": 2}\'')
    parser.add_argument(
        '--policies-to-train', nargs="+", type=str, default=["red"], # "blue",
        help="List of agent ids to train")
    parser.add_argument(
        '--training-scheme', type=str, default='CTDE',
        help="Can be either 'CTCE', 'DTDE' or 'CTDE'")

    args = parser.parse_args()
    # args.multiagent = {}
    # args.multiagent["policies_to_train"] = args.policies_to_train
    config = algorithm_config(**vars(args))
    # config.multiagent["policies_to_train"] =args.policies_to_train
    config.seed = args.seed
    config.callbacks(EvaluationCallbacks)
    config.environment(disable_env_checking=True) # FIXME 
    stop_conditions = {'timesteps_total': args.num_timesteps} 

    print()
    print(f"Running with following CLI options: {args}")
    print('\n', '-' * 64, '\n', "Training with following configuration:", '\n', '-' * 64)
    print()
    pprint(config.to_dict())
    train(algo=args.algo, config=config, stop_conditions=stop_conditions, save_dir=args.save_dir, load_dir=args.load_dir, local_mode=args.local_mode, experiment_name=args.name, training_scheme=args.training_scheme)
