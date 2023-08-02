from __future__ import annotations

import argparse
import json
import os
import ray

from multigrid.utils.training_utilis import algorithm_config, can_use_gpu,  get_checkpoint_dir, policy_mapping_fn
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
from ray.air.integrations.mlflow import MLflowLoggerCallback

import pathlib
SCRIPT_PATH = str(pathlib.Path(__file__).parent.absolute().parent.absolute())

import git

# Limit the number of rows.
reporter = CLIReporter(max_progress_rows=10)


tags = { "user_name" : "John",
         "git_commit_hash" : git.Repo(SCRIPT_PATH).head.commit
         }


# TODO - Set Evaluation

def train(
    algo: str,
    config: AlgorithmConfig,
    stop_conditions: dict,
    save_dir: str,
    load_dir: str | None = None,
    local_mode: bool = False,
    experiment_name: str = "testing_experiment",
    ):
    """
    Train an RLlib algorithm.
    """



    class RestoreWeightsCallback(DefaultCallbacks):

        def __init__(
                self,
                load_dir: str,
                policy_name: str,
            ):
            policy_0_checkpoint_path = get_checkpoint_dir(load_dir)
            restored_policy_0 = Policy.from_checkpoint(policy_0_checkpoint_path)
            self.restored_policy_0_weights = restored_policy_0[policy_name].get_weights()
            self.policy_name = policy_name


        def on_algorithm_init(self, *, algorithm: "Algorithm", **kwargs) -> None:
            algorithm.set_weights({self.policy_name: self.restored_policy_0_weights})




    ray.init(num_cpus=(config.num_rollout_workers + 1), local_mode=local_mode)
    tune.run(
        algo,
        stop=stop_conditions,
        config=config,
        local_dir=save_dir,
        verbose=1,
        restore=get_checkpoint_dir(load_dir),
        checkpoint_freq=5,
        checkpoint_at_end=True,
        progress_reporter=reporter,
        callbacks=[MLflowLoggerCallback(
        tracking_uri="./submission/mlflow",
        experiment_name=experiment_name,
            tags=tags,
            save_artifact=True)]
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
        '--env', type=str, default='MultiGrid-CompetativeRedBlueDoor-v0',
        help="MultiGrid environment to use.")
    parser.add_argument(
        '--env-config', type=json.loads, default={},
        help="Environment config dict, given as a JSON string (e.g. '{\"size\": 8}')")
    parser.add_argument(
        '--num-agents', type=int, default=2, help="Number of agents in environment.")
    parser.add_argument(
        '--seed', type=int, default=0, help="Set the random seed of each worker. This makes experiments reproducible")
    parser.add_argument(
        '--num-workers', type=int, default=6, help="Number of rollout workers.")
    parser.add_argument(
        '--num-gpus', type=int, default=0, help="Number of GPUs to train on.")
    parser.add_argument(
        '--num-timesteps', type=int, default=1e7,
        help="Total number of timesteps to train.")
    parser.add_argument(
        '--lr', type=float, help="Learning rate for training.")
    parser.add_argument(
        '--load-dir', type=str, default='/Users/zla0368/Documents/RL/RL_Class/code/multigrid/notebooks/submission/ray_results/PPO/PPO_MultiGrid-CompetativeRedBlueDoor-v0_52041_00000_0_2023-07-14_16-02-12',
        help="Checkpoint directory for loading pre-trained policies.")
    parser.add_argument(
        '--save-dir', type=str, default='submission/ray_results/',
        help="Directory for saving checkpoints, results, and trained policies.")
    parser.add_argument(
        '--name', type=str, default='<my_experinemnt>',
        help="Distinct name to track your experinemnt in save-dir")
    parser.add_argument(
        '--local-mode', type=bool, default=True,
        help="Boolean value to set to use local mode for debugging")
    parser.add_argument(
        '--our-agent-ids', nargs="+", type=int, default=[0,1],
        help="List of agent ids to train")
    parser.add_argument(
        '--policies-to-train', nargs="+", type=str, default=["policy_1"], # "policy_1",
        help="List of agent ids to train")


    args = parser.parse_args()
    # args.multiagent = {}
    # args.multiagent["policies_to_train"] = args.policies_to_train
    config = algorithm_config(**vars(args))
    # config.multiagent["policies_to_train"] =args.policies_to_train
    config.seed = args.seed
    stop_conditions = {'timesteps_total': args.num_timesteps}

    print()
    print(f"Running with following CLI options: {args}")
    print('\n', '-' * 64, '\n', "Training with following configuration:", '\n', '-' * 64)
    print()
    pprint(config.to_dict())
    train(algo=args.algo, config=config, stop_conditions=stop_conditions, save_dir=args.save_dir, load_dir=args.load_dir, local_mode=args.local_mode, experiment_name=args.name)
