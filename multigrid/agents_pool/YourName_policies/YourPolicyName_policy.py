import numpy as np
from multigrid.utils.policy import Policy
from multigrid.base import MultiGridEnv, AgentID, ObsType
from multigrid.envs import CONFIGURATIONS
from ray.rllib.utils.from_config import NotProvided
from ray.rllib.algorithms.ppo import PPOConfig
from gymnasium.core import ObservationWrapper

class YourPolicyName_Policy(Policy):
    """ 
        Policy class for Meltingpot competition 
        About Populations:
            We will make multiple instances of this class for every focal agent
            If you want to sample different agents for every population/episode
            add the required required randomization in the "initial_state" function
    """
    def __init__(self, policy_id:str , policy_name:str):
        # You can implement any init operations here or in setup()
        self.policy_id = policy_id # TODO - Should this be multiple or indiviaul, current is not individual
        self.policy_name = policy_name # TODO - Should this be multiple or indiviaul, current is not individual
        self.training_scheme = "DTDE"
        self.reward_schemes =    {
                                    self.policy_id: {
                                        "eliminated_opponent_sparse_reward": 0.5,
                                        "key_pickup_sparse_reward": 0.5,
                                        "ball_pickup_dense_reward": 0.5,
                                        "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                                        "invalid_pickup_dense_penalty": 0.001,
                                        }
                                }

        self.algorithm_training_config = {  
            self.policy_id: {
                    "algo": "PPO",
                    "algo_config_class" : PPOConfig,
                    "algo_config": {
                    "lr" : 0.0015, #NotProvided,
                    "gamma": 0.99,
                    "lambda_" : 0.99,
                    "kl_coeff" : 0.2,
                    "kl_target" : 0.01,
                    "clip_param" : 0.3,
                    "grad_clip" : None,
                    "vf_clip_param" : 10.0,
                    "vf_loss_coeff" : 0.5,            
                    "entropy_coeff" : 0.001,
                    "sgd_minibatch_size" : 128,
                    "num_sgd_iter" : 30,
                    }
            }
        }

    @staticmethod
    def custom_observations(obs: dict[AgentID, ObsType], agent_id: str, wrapper: ObservationWrapper):

        agent_observations = obs[agent_id]
        if isinstance(agent_observations, list):
            # If it is stacked observations from multiple agents
            for observation in agent_observations:
                # update the given ["image"] observation with self.one_hot() with the updated self.dim_sizes
                observation["image"] = wrapper.one_hot(observation["image"], wrapper.dim_sizes)
        else:
            # update the given ["image"] observation with self.one_hot() with the updated self.dim_sizes
            agent_observations["image"] = wrapper.one_hot(agent_observations["image"], wrapper.dim_sizes)

        return agent_observations

        




