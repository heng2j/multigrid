import numpy as np
from multigrid.utils.policy import Policy
from multigrid.envs import CONFIGURATIONS


class YourNamePolicy(Policy):
    """ 
        Policy class for Meltingpot competition 
        About Populations:
            We will make multiple instances of this class for every focal agent
            If you want to sample different agents for every population/episode
            add the required required randomization in the "initial_state" function
    """
    def __init__(self, policy_id):
        # You can implement any init operations here or in setup()
        self.policy_id = policy_id
        seed = 42
        self.rng = np.random.RandomState(seed)
        self.substrate_name = None
        self.reward_schemes =    {
                                    self.policy_id: {
                                        "eliminated_opponent_sparse_reward": 0.5,
                                        "key_pickup_sparse_reward": 0.5,
                                        "ball_pickup_dense_reward": 0.5,
                                        "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                                        "invalid_pickup_dense_penalty": 0.001,
                                        }
                                }



