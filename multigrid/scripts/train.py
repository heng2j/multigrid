""" Expected for restricted changes """

from __future__ import annotations

import argparse
import json
import os
import pathlib
import ray

from multigrid.utils.training_utilis import algorithm_config, get_checkpoint_dir, EvaluationCallbacks
from multigrid.rllib.ctde_torch_policy import CentralizedCritic

from pprint import pprint
from ray import tune
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.from_config import NotProvided
from ray.tune.registry import get_trainable_cls
from ray.tune import CLIReporter
from ray.air.integrations.mlflow import MLflowLoggerCallback


SCRIPT_PATH = str(pathlib.Path(__file__).parent.absolute().parent.absolute())

import git

# Limit the number of rows.
reporter = CLIReporter(max_progress_rows=10, max_report_frequency=30)


tags = {"user_name": "John", "git_commit_hash": git.Repo(SCRIPT_PATH).head.commit}

def train(
    algo: str,
    config: AlgorithmConfig,
    stop_conditions: dict,
    save_dir: str,
    load_dir: str | None = None,
    local_mode: bool = False,
    experiment_name: str = "testing_experiment",
    training_scheme: str = "CTDE",  # Can be either "CTCE", "DTDE" or "CTDE"
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
        callbacks=[
            MLflowLoggerCallback(
                tracking_uri="./submission/mlflow", experiment_name=experiment_name, tags=tags, save_artifact=True
            ),
        ],  #   RestoreWeightsCallback(load_dir=load_dir,policy_name="policy_0"),
    )
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="PPO", help="The name of the RLlib-registered algorithm to use.")
    parser.add_argument(
        "--framework", type=str, choices=["torch", "tf", "tf2"], default="torch", help="Deep learning framework to use."
    )
    parser.add_argument("--lstm", action="store_true", help="Use LSTM model.")
    parser.add_argument(
        "--env", type=str, default="MultiGrid-CompetativeRedBlueDoor-v3-CTDE-Red", help="MultiGrid environment to use."
    )
    parser.add_argument(
        "--env-config",
        type=json.loads,
        default={},
        help="Environment config dict, given as a JSON string (e.g. '{\"size\": 8}')",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Set the random seed of each worker. This makes experiments reproducible"
    )
    parser.add_argument("--num-workers", type=int, default=40, help="Number of rollout workers.")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to train on.")
    parser.add_argument("--num-timesteps", type=int, default=1e6, help="Total number of timesteps to train.")
    parser.add_argument("--lr", type=float, help="Learning rate for training.")
    parser.add_argument(
        "--load-dir",
        type=str,  # default='/Users/zla0368/Documents/RL/RL_Class/code/multigrid/submission/ray_results/PPO/PPO_MultiGrid-CompetativeRedBlueDoor-v0_5dfa1_00000_0_2023-08-01_23-12-09',
        help="Checkpoint directory for loading pre-trained policies.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="submission/ray_results/",
        help="Directory for saving checkpoints, results, and trained policies.",
    )
    parser.add_argument(
        "--name", type=str, default="<my_experinemnt>", help="Distinct name to track your experinemnt in save-dir"
    )
    parser.add_argument(
        "--local-mode", type=bool, default=False, help="Boolean value to set to use local mode for debugging"
    )
    parser.add_argument("--our-agent-ids", nargs="+", type=int, default=[0, 1], help="List of agent ids to train")
    parser.add_argument(
        "--policies-to-train", nargs="+", type=str, default=["red"], help="List of agent ids to train"  # "blue",
    )
    parser.add_argument("--training-scheme", type=str, default="CTDE", help="Can be either 'CTCE', 'DTDE' or 'CTDE'")

    args = parser.parse_args()
    # args.multiagent = {}
    # args.multiagent["policies_to_train"] = args.policies_to_train
    config = algorithm_config(**vars(args))
    # config.multiagent["policies_to_train"] =args.policies_to_train
    config.seed = args.seed
    config.callbacks(EvaluationCallbacks)
    config.environment(disable_env_checking=False)
    stop_conditions = {"timesteps_total": args.num_timesteps}

    print()
    print(f"Running with following CLI options: {args}")
    print("\n", "-" * 64, "\n", "Training with following configuration:", "\n", "-" * 64)
    print()
    pprint(config.to_dict())
    train(
        algo=args.algo,
        config=config,
        stop_conditions=stop_conditions,
        save_dir=args.save_dir,
        load_dir=args.load_dir,
        local_mode=args.local_mode,
        experiment_name=args.name,
        training_scheme=args.training_scheme,
    )
