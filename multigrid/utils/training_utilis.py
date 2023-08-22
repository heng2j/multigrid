import os
from pathlib import Path
import itertools
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.algorithms import AlgorithmConfig
from multigrid.rllib.models import TFModel, TorchModel, TorchLSTMModel, TorchCentralizedCriticModel
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



def policy_mapping_fn(training_scheme:str, teams: dict[str, int], agent_index_dict: dict[int, str]):

    # FIXME
    def single_policy_mapping_fn(agent_id, *args,  **kwargs) -> str:
        """
        Map an environment agent ID to an RLlib policy ID.
        """
        return   agent_id #agent_index_dict[agent_id] # f'policy_{agent_id}'

    def marl_policy_mapping_fn(agent_id, *args, **kwargs) -> str:
        """
        Map an environment agent ID to an RLlib policy ID.
        """
        if training_scheme == "DTDE" or "CTDE":
            return agent_id


    return  single_policy_mapping_fn  if training_scheme == "CTCE" else marl_policy_mapping_fn




   

def model_config(
    framework: str = 'torch',
    lstm: bool = False,
    custom_model_config: dict = {}
    ):
    """
    Return a model configuration dictionary for RLlib.
    """
    if framework == 'torch':
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


# TODO - Set CTDE Model
# https://github.com/ray-project/ray/blob/master/rllib/examples/centralized_critic.py
# https://github.com/ray-project/ray/blob/master/rllib/examples/models/centralized_critic_models.py


import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch

from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing
from ray.rllib.algorithms.ppo.ppo import PPO, PPOConfig

from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.complex_input_net import (
    ComplexInputNetwork as TorchComplexInputNetwork
)
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import convert_to_torch_tensor

OPPONENT_OBS = "opponent_obs"
OPPONENT_ACTION = "opponent_action"




# ModelCatalog.register_custom_model(
#     "CTDE_model",
#     TorchCentralizedCriticModel
# )

# class CentralizedValueMixin:
#     """Add method to evaluate the central value function from the model."""

#     def __init__(self):
#         self.compute_central_vf = self.model.central_value_function



# # Grabs the opponent obs/act and includes it in the experience train_batch,
# # and computes GAE using the central vf predictions.
# def centralized_critic_postprocessing(
#     policy, sample_batch, other_agent_batches=None, episode=None
# ):
#     pytorch = policy.config["framework"] == "torch"
#     if (pytorch and hasattr(policy, "compute_central_vf")) or (
#         not pytorch and policy.loss_initialized()
#     ):
#         assert other_agent_batches is not None
#         if policy.config["enable_connectors"]:
#             [(_, _, opponent_batch)] = list(other_agent_batches.values())
#         else:
#             [(_, opponent_batch)] = list(other_agent_batches.values())

#         # also record the opponent obs and actions in the trajectory
#         sample_batch[OPPONENT_OBS] = opponent_batch[SampleBatch.CUR_OBS]
#         sample_batch[OPPONENT_ACTION] = opponent_batch[SampleBatch.ACTIONS]

#         # overwrite default VF prediction with the central VF
#         sample_batch[SampleBatch.VF_PREDS] = (
#             policy.compute_central_vf(
#                 convert_to_torch_tensor(
#                     sample_batch[SampleBatch.CUR_OBS], policy.device
#                 ),
#                 convert_to_torch_tensor(sample_batch[OPPONENT_OBS], policy.device),
#                 convert_to_torch_tensor(
#                     sample_batch[OPPONENT_ACTION], policy.device
#                 ),
#             )
#             .cpu()
#             .detach()
#             .numpy()
#         )

#     else:
#         # Policy hasn't been initialized yet, use zeros.
#         sample_batch[OPPONENT_OBS] = np.zeros_like(sample_batch[SampleBatch.CUR_OBS])
#         sample_batch[OPPONENT_ACTION] = np.zeros_like(sample_batch[SampleBatch.ACTIONS])
#         sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(
#             sample_batch[SampleBatch.REWARDS], dtype=np.float32
#         )

#     completed = sample_batch[SampleBatch.TERMINATEDS][-1]
#     if completed:
#         last_r = 0.0
#     else:
#         last_r = sample_batch[SampleBatch.VF_PREDS][-1]

#     train_batch = compute_advantages(
#         sample_batch,
#         last_r,
#         policy.config["gamma"],
#         policy.config["lambda"],
#         use_gae=policy.config["use_gae"],
#     )
#     return train_batch


# # Copied from PPO but optimizing the central value function.
# def loss_with_central_critic(policy, base_policy, model, dist_class, train_batch):
#     # Save original value function.
#     vf_saved = model.value_function

#     # Calculate loss with a custom value function.
#     model.value_function = lambda: policy.model.central_value_function(
#         train_batch[SampleBatch.CUR_OBS],
#         train_batch[OPPONENT_OBS],
#         train_batch[OPPONENT_ACTION],
#     )
#     policy._central_value_out = model.value_function()
#     loss = base_policy.loss(model, dist_class, train_batch)

#     # Restore original value function.
#     model.value_function = vf_saved

#     return loss


# class CTDEPPOTorchPolicy(CentralizedValueMixin, PPOTorchPolicy):
#     def __init__(self, observation_space, action_space, config):
#         PPOTorchPolicy.__init__(self, observation_space, action_space, config)
#         CentralizedValueMixin.__init__(self)

#     @override(PPOTorchPolicy)
#     def loss(self, model, dist_class, train_batch):
#         return loss_with_central_critic(self, super(), model, dist_class, train_batch)

#     @override(PPOTorchPolicy)
#     def postprocess_trajectory(
#         self, sample_batch, other_agent_batches=None, episode=None
#     ):
#         return centralized_critic_postprocessing(
#             self, sample_batch, other_agent_batches, episode
#         )


# class CentralizedCritic(PPO):
#     @classmethod
#     @override(PPO)
#     def get_default_policy_class(cls, config):
#         return CTDEPPOTorchPolicy




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
    training_scheme: str = "CTCE", # Can be either "CTCE", "DTDE" or "CTDE"  
    **kwargs) -> AlgorithmConfig:
    """
    Return the RL algorithm configuration dictionary.
    """
    env_config = {**env_config, 'agents': num_agents, "teams": teams, "training_scheme": training_scheme}
    agent_index_dict = {agent_id: next(team for team, count in teams.items() if sum(teams[t] for t in itertools.takewhile(lambda x: x != team, teams)) + count > agent_id) for agent_id in range(num_agents)}
    return (
        get_trainable_cls(algo) # CentralizedCritic #FIXME if training_scheme == "CTDE" else 
        .get_default_config()
        .environment(env=env, env_config=env_config)
        .framework(framework)
        .rollouts(num_rollout_workers=num_workers)
        .resources(num_gpus=num_gpus) #if can_use_gpu() else 0)
        .multi_agent(
            # policies={f'policy_{i}' for i in our_agent_ids},
            policies={ team_name for team_name in list(teams.keys())} if training_scheme == "CTCE" else {f'{team_name}_{i}' for team_name, team_num in teams.items() for i in range(team_num)},
            policy_mapping_fn=policy_mapping_fn(training_scheme=training_scheme,teams=teams, agent_index_dict=agent_index_dict),
            # policies_to_train=policies_to_train
        )
        .training(
            model=model_config(framework=framework, lstm=lstm, custom_model_config={ "teams": teams, "training_scheme": training_scheme}),
            lr=(lr or NotProvided),
        )
    )