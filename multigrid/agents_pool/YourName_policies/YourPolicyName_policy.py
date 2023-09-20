import numpy as np
from multigrid.utils.policy import Policy
from multigrid.base import MultiGridEnv, AgentID, ObsType
from multigrid.core.agent import Agent, Mission
from multigrid.core import Action
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

        
    @staticmethod
    def custom_handle_steps( agent, agent_index, action, reward, terminated, info, env):
        
        fwd_obj = env.grid.get(*agent.front_pos)

        if action == Action.toggle:
            for other_agent in env.agents:
                if (agent.front_pos == other_agent.pos) and other_agent.color != agent.color:
                    fwd_obj = other_agent

            # If fwd_obj is a door
            if fwd_obj == env.red_door or fwd_obj == env.blue_door:
                if (env.red_door.is_open or env.blue_door.is_open) and (fwd_obj.color == agent.color):
                    # Set Done Conditions for winning team
                    for this_agent in env.agents:
                        if this_agent.color == agent.color and not this_agent.terminated:
                            # this_agent.mission = Mission("We won!")
                            env.on_success(
                                this_agent, reward, terminated
                            )  # reward the rest of the teammembers who are still standing
                            info[this_agent.color if env.training_scheme == "CTCE" else this_agent.name][
                                "door_open_done"
                            ] = True

                    # self.info["episode_done"].get("l", self.step_count)

            # If fwd_obj is an agent
            elif isinstance(fwd_obj, Agent) and env.death_match:
                # Terminate the other agent and set it's position inside the room
                # fwd_obj.mission = Mission("I died!")
                env.on_failure(fwd_obj, reward)
                info[fwd_obj.color if env.training_scheme == "CTCE" else fwd_obj.name]["got_eliminated_done"] = True
                env.grid.set(*fwd_obj.pos, None)
                fwd_obj.pos = (
                    (13, 2) if fwd_obj.color == "blue" else (2, 2)
                )  # NOTE This is not scalabe and only works in 2v2 at most
                reward[agent_index] += env.reward_schemes[agent.name]["eliminated_opponent_sparse_reward"]
                reward[fwd_obj.index] -= 1  # NOTE - This opponent penalty is a fixed value for the game

                # Terminate the game if the rest of the other agents in the same team also got terminated
                all_opponents_terminated = all(
                    [other_agent.terminated for other_agent in env.agents if other_agent.color != agent.color]
                )
                if all_opponents_terminated:
                    for this_agent in env.agents:
                        if this_agent.color == agent.color and not this_agent.terminated:
                            env.on_success(this_agent, reward, terminated)
                            info[this_agent.color if env.training_scheme == "CTCE" else this_agent.name][
                                "eliminated_opponents_done"
                            ] = True
                            info[this_agent.color if env.training_scheme == "CTCE" else this_agent.name][
                                "eliminated_opponent_num"
                            ] += 1

        elif action == Action.pickup:
            if (
                agent.carrying
                and (agent.carrying.type == "key")
                and (agent.carrying.is_available)
                and (agent.color == agent.carrying.color)
            ):
                agent.carrying.is_available = False
                agent.carrying.is_pickedup = True
                reward[agent_index] += env.reward_schemes[agent.name]["key_pickup_sparse_reward"]

                if env.training_scheme == "DTDE" or "CTDE":
                    # Mimic communiations
                    agent.mission = Mission("Go open the door with the key")
                    for this_agent in env.agents:
                        if (this_agent.color == agent.color) and (this_agent != agent):
                            this_agent.mission = Mission("Go move away the ball")

            elif (
                agent.carrying
                and (agent.carrying.type == "ball")
                and (agent.front_pos == agent.carrying.init_pos)
                and (agent.color != agent.carrying.color)
            ):
                reward[agent_index] += (
                    env.reward_schemes[agent.name]["ball_pickup_dense_reward"] * agent.carrying.discount_factor
                )
                agent.carrying.discount_factor *= agent.carrying.discount_factor

                if env.training_scheme == "DTDE" or "CTDE":
                    # Mimic communiations
                    agent.mission = Mission("Go move away the ball")
                    for this_agent in env.agents:
                        if (this_agent.color == agent.color) and (this_agent != agent):
                            if (
                                this_agent.carrying
                                and this_agent.carrying.type == "key"
                                and this_agent.carrying.color == this_agent.color
                            ):
                                this_agent.mission = Mission("Go open the door with the key")
                            else:
                                this_agent.mission = Mission("Go pick up the key")

            else:
                # Invalid pickup action
                reward[agent_index] -= env.reward_schemes[agent.name]["invalid_pickup_dense_penalty"]



