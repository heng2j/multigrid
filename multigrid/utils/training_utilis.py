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
                framework=framework, lstm=lstm, custom_model_config={"teams": env_config["teams"], "training_scheme": env_config["training_scheme"]}
            ),
            lr=(lr or NotProvided),
        )
    )
