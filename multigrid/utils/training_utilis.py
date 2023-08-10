import os
from pathlib import Path
import itertools
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.algorithms import AlgorithmConfig
from multigrid.rllib.models import TFModel, TorchModel, TorchLSTMModel
from ray.rllib.utils.from_config import NotProvided
from ray.tune.registry import get_trainable_cls



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
        checkpoints = Path(search_dir).expanduser().glob('**/*.is_checkpoint')
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



def policy_mapping_fn(trianing_scheme:str, teams: dict[str, int], agent_index_dict: dict[int, str]):

    def single_policy_mapping_fn(agent_id, *args,  **kwargs) -> str:
        """
        Map an environment agent ID to an RLlib policy ID.
        """
        return   agent_id #agent_index_dict[agent_id] #f'policy_{agent_id}'

    def marl_policy_mapping_fn(agent_id, *args, **kwargs) -> str:
        """
        Map an environment agent ID to an RLlib policy ID.
        """
        if trianing_scheme == "DTDE":
            ...
        elif trianing_scheme == "CTDE":
            ...


   

    return  single_policy_mapping_fn  if trianing_scheme == "CTCE" else marl_policy_mapping_fn




   

def model_config(
    framework: str = 'torch',
    lstm: bool = False,
    custom_model_config: dict = {}):
    """
    Return a model configuration dictionary for RLlib.
    """
    if framework == 'torch':
        if lstm:
            model = TorchLSTMModel
        else:
            model = TorchModel
    else:
        if lstm:
            raise NotImplementedError
        else:
            model = TFModel

    return {
        'custom_model': model,
        'custom_model_config': custom_model_config,
        'conv_filters': [
            [16, [3, 3], 1],
            [32, [3, 3], 1],
            [64, [3, 3], 1],
        ],
        'fcnet_hiddens': [64, 64],
        'post_fcnet_hiddens': [],
        'lstm_cell_size': 256,
        'max_seq_len': 20,
    }

def algorithm_config(
    algo: str = 'PPO',
    env: str = 'MultiGrid-Empty-8x8-v0',
    env_config: dict = {},
    num_agents: int = 2,
    framework: str = 'torch',
    lstm: bool = False,
    num_workers: int = 0,
    num_gpus: int = 0,
    lr: float | None = None,
    policies_to_train: list[int] | None = None,
    our_agent_ids: list[str] | None = None,
    teams: dict[str, int] = {"red": 1},
    trianing_scheme: str = "CTCE", # Can be either "CTCE", "DTDE" or "CTDE"  # FIXME - this need to be coming from argument parser or config
    **kwargs) -> AlgorithmConfig:
    """
    Return the RL algorithm configuration dictionary.
    """
    env_config = {**env_config, 'agents': num_agents}
    agent_index_dict = {agent_id: next(team for team, count in teams.items() if sum(teams[t] for t in itertools.takewhile(lambda x: x != team, teams)) + count > agent_id) for agent_id in range(num_agents)}
    return (
        get_trainable_cls(algo)
        .get_default_config()
        .environment(env=env, env_config=env_config)
        .framework(framework)
        .rollouts(num_rollout_workers=num_workers)
        .resources(num_gpus=num_gpus if can_use_gpu() else 0)
        .multi_agent(
            # policies={f'policy_{i}' for i in our_agent_ids},
            policies={ team_name for team_name in list(teams.keys())},
            policy_mapping_fn=policy_mapping_fn(trianing_scheme=trianing_scheme,teams=teams, agent_index_dict=agent_index_dict),
            # policies_to_train=policies_to_train
        )
        .training(
            model=model_config(framework=framework, lstm=lstm),
            lr=(lr or NotProvided),
        )
    )