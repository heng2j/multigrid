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



def train(
    algo: str,
    config: AlgorithmConfig,
    stop_conditions: dict,
    save_dir: str,
    load_dir: str | None = None):
    """
    Train an RLlib algorithm.
    """
    ray.init(num_cpus=(config.num_rollout_workers + 1))
    tune.run(
        algo,
        stop=stop_conditions,
        config=config,
        local_dir=save_dir,
        verbose=1,
        restore=get_checkpoint_dir(load_dir),
        checkpoint_freq=20,
        checkpoint_at_end=True,
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
        '--load-dir', type=str,
        help="Checkpoint directory for loading pre-trained policies.")
    parser.add_argument(
        '--save-dir', type=str, default='./ray_results/',
        help="Directory for saving checkpoints, results, and trained policies.")
    parser.add_argument(
        '--name', type=str, default='<my_experinemnt>',
        help="Distinct name to track your experinemnt in save-dir")

    args = parser.parse_args()
    config = algorithm_config(**vars(args))
    config.seed = args.seed
    stop_conditions = {'timesteps_total': args.num_timesteps}

    print()
    print(f"Running with following CLI options: {args}")
    print('\n', '-' * 64, '\n', "Training with following configuration:", '\n', '-' * 64)
    print()
    pprint(config.to_dict())
    train(args.algo, config, stop_conditions, args.save_dir, args.load_dir)
