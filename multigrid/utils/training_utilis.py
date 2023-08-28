""" Expected for restricted changes """

import os
from pathlib import Path
import itertools
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.algorithms import AlgorithmConfig
from multigrid.rllib.models import TFModel, TorchModel, TorchLSTMModel, TorchCentralizedCriticModel
from ray.rllib.utils.from_config import NotProvided
from ray.tune.registry import get_trainable_cls
from gymnasium.envs import registry as gym_envs_registry
import ray.rllib.algorithms.callbacks as callbacks
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.callback import Callback
from ray.rllib import BaseEnv, Policy, RolloutWorker
from typing import Dict, Optional, Union
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import AgentID, EnvType, PolicyID


def get_checkpoint_dir(search_dir: Path | str | None) -> Path | None:
    """
    Recursively search for checkpoints within the given directory.

    If more than one is found, returns the most recently modified checkpoint directory.

    Parameters
    ----------
    search_dir : Path or str
        The directory to search for checkpoints within
    """
    if search_dir:
        checkpoints = Path(search_dir).expanduser().glob("**/*.is_checkpoint")
        if checkpoints:
            return sorted(checkpoints, key=os.path.getmtime)[-1].parent

    return None


def can_use_gpu() -> bool:
    """
    Return whether or not GPU training is available.
    """
    try:
        _, tf, _ = try_import_tf()
        return tf.test.is_gpu_available()
    except:
        pass

    try:
        torch, _ = try_import_torch()
        return torch.cuda.is_available()
    except:
        pass

    return False


def model_config(framework: str = "torch", lstm: bool = False, custom_model_config: dict = {}):
    """
    Return a model configuration dictionary for RLlib.
    """
    if framework == "torch":
        if lstm:
            model = TorchLSTMModel
        elif custom_model_config["training_scheme"] == "CTDE":
            model = TorchCentralizedCriticModel
        else:
            model = TorchModel
    else:
        if lstm:
            raise NotImplementedError
        else:
            model = TFModel

    return {
        "custom_model": model,
        "custom_model_config": custom_model_config,
        "conv_filters": [
            [16, [3, 3], 1],
            [32, [3, 3], 1],
            [64, [3, 3], 1],
        ],
        "fcnet_hiddens": [64, 64],
        "post_fcnet_hiddens": [],
        "lstm_cell_size": 256,
        "max_seq_len": 20,
    }


def algorithm_config(
    algo: str = "PPO",
    env: str = "MultiGrid-Empty-8x8-v0",
    env_config: dict = {},
    num_agents: int = 2,
    framework: str = "torch",
    lstm: bool = False,
    num_workers: int = 0,
    num_gpus: int = 0,
    lr: float | None = None,
    policies_to_train: list[int] | None = None,
    our_agent_ids: list[str] | None = None,
    **kwargs,
) -> AlgorithmConfig:
    """
    Return the RL algorithm configuration dictionary.
    """

    env_config = gym_envs_registry[env].kwargs

    return (
        get_trainable_cls(algo)
        .get_default_config()
        .environment(env=env, env_config=env_config)
        .framework(framework)
        .rollouts(num_rollout_workers=num_workers)
        .resources(num_gpus=num_gpus)
        .multi_agent(
            policies={team_name for team_name in list(env_config["teams"].keys())}
            if env_config["training_scheme"] == "CTCE"
            else {f"{team_name}_{i}" for team_name, team_num in env_config["teams"].items() for i in range(team_num)},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
        )
        .training(
            model=model_config(
                framework=framework,
                lstm=lstm,
                custom_model_config={"teams": env_config["teams"], "training_scheme": env_config["training_scheme"]},
            ),
            lr=(lr or NotProvided),
        )
    )



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
