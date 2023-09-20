"""Template for custom policy implementations."""

import abc
from typing import Generic, Tuple, TypeVar
from gymnasium.core import ObservationWrapper
from multigrid.base import AgentID, ObsType

State = TypeVar('State')


class Policy(Generic[State], metaclass=abc.ABCMeta):
  """Abstract base class for a policy.
  """
  def __init__(self, policy_id:str , policy_name:str):
      # You can implement any init operations here or in setup()
      self.policy_id = policy_id # TODO - Should this be multiple or indiviaul, current is not individual
      self.policy_name = policy_name # TODO - Should this be multiple or indiviaul, current is not individual
      self.training_scheme = "DTDE" # Can be either "CTCE", "DTDE" or "CTDE"
      self.reward_schemes =  { self.policy_id: {}}
      self.algorithm_training_config = {self.policy_id: {}}


  @abc.abstractmethod
  @staticmethod
  def custom_observations(obs: dict[AgentID, ObsType], agent_id: str, wrapper: ObservationWrapper):
    raise NotImplementedError()

  @abc.abstractmethod
  @staticmethod
  def custom_handle_steps( agent, agent_index, action, reward, terminated, info, env):
    raise NotImplementedError()



  def __enter__(self):
    return self

  def __exit__(self, *args, **kwargs):
    del args, kwargs
    self.close()