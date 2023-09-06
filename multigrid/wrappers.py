from __future__ import annotations


import numba as nb
import numpy as np
from numpy.typing import NDArray as ndarray
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ObservationWrapper

from .base import MultiGridEnv, AgentID, ObsType
from .core.constants import Color, Direction, State, Type
from .core.world_object import WorldObj


class FullyObsWrapper(ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding instead of agent view.

    Examples
    --------
    >>> import gymnasium as gym
    >>> import multigrid.envs
    >>> env = gym.make('MultiGrid-Empty-16x16-v0')
    >>> obs, _ = env.reset()
    >>> obs[0]['image'].shape
    (7, 7, 3)

    >>> from multigrid.wrappers import FullyObsWrapper
    >>> env = FullyObsWrapper(env)
    >>> obs, _ = env.reset()
    >>> obs[0]['image'].shape
    (16, 16, 3)
    """

    def __init__(self, env: MultiGridEnv):
        """ """
        super().__init__(env)

        # Update agent observation spaces
        for agent in self.env.agents:
            agent.observation_space["image"] = spaces.Box(
                low=0, high=255, shape=(env.height, env.width, WorldObj.dim), dtype=int
            )

    def observation(self, obs: dict[AgentID, ObsType]) -> dict[AgentID, ObsType]:
        """
        :meta private:
        """
        img = self.env.grid.encode()
        for agent in self.env.agents:
            img[agent.state.pos] = agent.encode()

        for agent_id in obs:
            obs[agent_id]["image"] = img

        return obs


class OneHotObsWrapper(ObservationWrapper):
    """
    Wrapper to get a one-hot encoding of a partially observable
    agent view as observation.

    Examples
    --------
    >>> from multigrid.envs import EmptyEnv
    >>> from multigrid.wrappers import OneHotObsWrapper
    >>> env = EmptyEnv()
    >>> obs, _ = env.reset()
    >>> obs[0]['image'][0, :, :]
    array([[2, 5, 0],
            [2, 5, 0],
            [2, 5, 0],
            [2, 5, 0],
            [2, 5, 0],
            [2, 5, 0],
            [2, 5, 0]])
    >>> env = OneHotObsWrapper(env)
    >>> obs, _ = env.reset()
    >>> obs[0]['image'][0, :, :]
    array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]],
            dtype=uint8)
    """

    def __init__(self, env: MultiGridEnv):
        """ """
        super().__init__(env)
        self.dim_sizes = np.array([len(Type), len(Color), max(len(State), len(Direction))])

        # Update agent observation spaces
        dim = sum(self.dim_sizes)
        for agent in self.env.agents:
            view_height, view_width, _ = agent.observation_space["image"].shape
            agent.observation_space["image"] = spaces.Box(
                low=0, high=1, shape=(view_height, view_width, dim), dtype=np.uint8
            )

    def observation(self, obs: dict[AgentID, ObsType]) -> dict[AgentID, ObsType]:
        """
        :meta private:
        """
        for agent_id in obs:
            obs[agent_id]["image"] = self.one_hot(obs[agent_id]["image"], self.dim_sizes)

        return obs

    @staticmethod
    @nb.njit(cache=True)
    def one_hot(x: ndarray[np.int], dim_sizes: ndarray[np.int]) -> ndarray[np.uint8]:
        """
        Return a one-hot encoding of a 3D integer array,
        where each 2D slice is encoded separately.

        Parameters
        ----------
        x : ndarray[int] of shape (view_height, view_width, dim)
            3D array of integers to be one-hot encoded
        dim_sizes : ndarray[int] of shape (dim,)
            Number of possible values for each dimension

        Returns
        -------
        out : ndarray[uint8] of shape (view_height, view_width, sum(dim_sizes))
            One-hot encoding

        :meta private:
        """
        out = np.zeros((x.shape[0], x.shape[1], sum(dim_sizes)), dtype=np.uint8)

        dim_offset = 0
        for d in range(len(dim_sizes)):
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    k = dim_offset + x[i, j, d]
                    out[i, j, k] = 1

            dim_offset += dim_sizes[d]

        return out


class SingleAgentWrapper(gym.Wrapper):
    """
    Wrapper to convert a multi-agent environment into a
    single-agent environment.
    """

    def __init__(self, env: MultiGridEnv):
        """ """
        super().__init__(env)
        self.observation_space = env.agents[0].observation_space
        self.action_space = env.agents[0].action_space


    def reset(self, *args, **kwargs):
        """
        :meta private:
        """
        result = super().reset(*args, **kwargs)
        return tuple(item for item in result)

    def step(self, action):
        """
        :meta private:
        """
        result = super().step({self.agents[0].name: action})
        return tuple(item for item in result)


class CompetativeRedBlueDoorWrapper(ObservationWrapper):
    """
    Wrapper to get a one-hot encoding of a partially observable
    agent view as observation.

    Examples
    --------
    >>> from multigrid.envs import EmptyEnv
    >>> from multigrid.wrappers import CompetativeRedBlueDoorWrapper
    >>> env = EmptyEnv()
    >>> obs, _ = env.reset()
    >>> obs[0]['image'][0, :, :]
    array([[2, 5, 0],
            [2, 5, 0],
            [2, 5, 0],
            [2, 5, 0],
            [2, 5, 0],
            [2, 5, 0],
            [2, 5, 0]])
    >>> env = CompetativeRedBlueDoorWrapper(env)
    >>> obs, _ = env.reset()
    >>> obs[0]['image'][0, :, :]
    array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]],
            dtype=uint8)
    """

    # NOTE - Questions
    def __init__(self, env: MultiGridEnv):
        """ """
        super().__init__(env)
        self.script_path = __file__

        # HW1 TODO 1:
        # Instead of directly using the RGB 3 channels  partially observable agent view 
        # In this wrapper, we are applying one-hot encoding of a partially observable agent view 
        # using Type, Color, State and Direction
        self.dim_sizes = np.array([len(Type), len(Color), max(len(State), len(Direction))])

        # Update agent observation spaces
        dim = sum(self.dim_sizes)
        for agent in self.env.agents:
            # Retrieve the shape of the original "image" observation_space 
            view_height, view_width, _ = agent.observation_space["image"].shape
            # Reassign the "image" observation_space for one-hot encoding observations
            agent.observation_space["image"] = spaces.Box(
                low=0, high=1, shape=(view_height, view_width, dim), dtype=np.uint8
            )

    def observation(self, obs: dict[AgentID, ObsType]) -> dict[AgentID, ObsType]:
        """
        :meta private:
        """
        # HW1 TODO 2:
        # For each agent_id in obs, update obs[agent_id]['image'] using the self.one_hot() method and 'image' from obs[agent_id].
        # If there's a type mismatch or one of the sub-observations is out of bounds, you might encounter an error like this:
        # ValueError: The observation collected from env.reset was not contained within your env's observation space.
        #             Its possible that there was a typemismatch (for example observations of np.float32 and a space ofnp.float64 observations),
        #             or that one of the sub-observations wasout of bounds.
        # Make sure to handle this exception and implement the correct observation to avoid it.

        for agent_id in obs:
            agent_observations = obs[agent_id]
            if isinstance(agent_observations, list):
                # If it is stacked observations from multiple agents
                for observation in agent_observations:
                    # update the given ["image"] observation with self.one_hot() with the updated self.dim_sizes
                    observation["image"] = self.one_hot(observation["image"], self.dim_sizes)
            else:
                # update the given ["image"] observation with self.one_hot() with the updated self.dim_sizes
                agent_observations["image"] = self.one_hot(agent_observations["image"], self.dim_sizes)
        
        return obs


    @staticmethod
    @nb.njit(cache=True)
    def one_hot(x: ndarray[np.int], dim_sizes: ndarray[np.int]) -> ndarray[np.uint8]:
        """
        Return a one-hot encoding of a 3D integer array,
        where each 2D slice is encoded separately.

        Parameters
        ----------
        x : ndarray[int] of shape (view_height, view_width, dim)
            3D array of integers to be one-hot encoded
        dim_sizes : ndarray[int] of shape (dim,)
            Number of possible values for each dimension

        Returns
        -------
        out : ndarray[uint8] of shape (view_height, view_width, sum(dim_sizes))
            One-hot encoding

        :meta private:
        """
        out = np.zeros((x.shape[0], x.shape[1], sum(dim_sizes)), dtype=np.uint8)

        dim_offset = 0
        for d in range(len(dim_sizes)):
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    k = dim_offset + x[i, j, d]
                    out[i, j, k] = 1

            dim_offset += dim_sizes[d]

        return out


class SingleAgentWrapperV2(gym.Wrapper):
    """
    Wrapper to convert a multi-agent environment into a
    single-agent environment.
    """

    def __init__(self, env: MultiGridEnv):
        """ """
        super().__init__(env)
        self.observation_space = env.agents[0].observation_space["image"]
        self.action_space = env.agents[0].action_space

        # self.observation_space

    def reset(self, *args, **kwargs):
        """
        :meta private:
        """
        result = super().reset(*args, **kwargs)
        return tuple(item for item in result)

    def step(self, action):
        """
        :meta private:
        """
        result = super().step({self.agents[0].name: action})
        return tuple(item for item in result)

class CompetativeRedBlueDoorWrapperV2(ObservationWrapper):
    """
    Wrapper to get a one-hot encoding of a partially observable
    agent view as observation.

    Examples
    --------
    >>> from multigrid.envs import EmptyEnv
    >>> from multigrid.wrappers import CompetativeRedBlueDoorWrapper
    >>> env = EmptyEnv()
    >>> obs, _ = env.reset()
    >>> obs[0]['image'][0, :, :]
    array([[2, 5, 0],
            [2, 5, 0],
            [2, 5, 0],
            [2, 5, 0],
            [2, 5, 0],
            [2, 5, 0],
            [2, 5, 0]])
    >>> env = CompetativeRedBlueDoorWrapper(env)
    >>> obs, _ = env.reset()
    >>> obs[0]['image'][0, :, :]
    array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]],
            dtype=uint8)
    """

    # NOTE - Questions
    def __init__(self, env: MultiGridEnv):
        """ """
        super().__init__(env)
        self.script_path = __file__

        # HW1 TODO 1:
        # Instead of directly using the RGB 3 channels  partially observable agent view 
        # In this wrapper, we are applying one-hot encoding of a partially observable agent view 
        # using Type, Color, State and Direction
        self.dim_sizes = np.array([len(Type), len(Color), max(len(State), len(Direction))])

        # Update agent observation spaces
        dim = sum(self.dim_sizes)
        for agent in self.env.agents:
            # Retrieve the shape of the original "image" observation_space 
            view_height, view_width, _ = agent.observation_space["image"].shape
            # Reassign the "image" observation_space for one-hot encoding observations
            agent.observation_space["image"] = spaces.Box(
                low=0, high=1, shape=(view_height, view_width, dim), dtype=np.uint8
            )

            # agent_space_dict = agent.observation_space.spaces.copy()
            # agent_space_dict.pop('mission')
            # agent.observation_space = spaces.Dict(agent_space_dict)
            # agent.original_observation_space = agent.observation_space.spaces.copy()
            # agent.observation_space = self.convert_dict_space_to_single_space(agent.observation_space)


        # self.observation_space = self.env.agents[0].observation_space
        self.observation_space = self.env.agents[0].observation_space["image"] 



    def observation(self, obs: dict[AgentID, ObsType]) -> dict[AgentID, ObsType]:
        """
        :meta private:
        """
        # HW1 TODO 2:
        # For each agent_id in obs, update obs[agent_id]['image'] using the self.one_hot() method and 'image' from obs[agent_id].
        # If there's a type mismatch or one of the sub-observations is out of bounds, you might encounter an error like this:
        # ValueError: The observation collected from env.reset was not contained within your env's observation space.
        #             Its possible that there was a typemismatch (for example observations of np.float32 and a space ofnp.float64 observations),
        #             or that one of the sub-observations wasout of bounds.
        # Make sure to handle this exception and implement the correct observation to avoid it.

        agent_id = list(obs.keys())[0]
        for agent_id in obs:
            agent_observations = obs[agent_id]
            if isinstance(agent_observations, list):
                # If it is stacked observations from multiple agents
                for observation in agent_observations:
                    # update the given ["image"] observation with self.one_hot() with the updated self.dim_sizes
                    observation["image"] = self.one_hot(observation["image"], self.dim_sizes)
            else:
                # update the given ["image"] observation with self.one_hot() with the updated self.dim_sizes
                agent_observations["image"] = self.one_hot(agent_observations["image"], self.dim_sizes)
        
        # return self.convert_dict_obs_to_single_obs(obs_dict=obs,obs_space=self.observation_space, action_space=self.action_space) 
        return obs[agent_id]["image"]

    @staticmethod
    def convert_dict_space_to_single_space(dict_space: spaces.Dict) -> spaces.Box:
        total_size = 0
        
        # Loop over each item in the dictionary
        for key, space in dict_space.spaces.items():
            if isinstance(space, spaces.Discrete):
                total_size += space.n
            elif isinstance(space, spaces.Box):
                total_size += np.prod(space.shape)
            else:
                raise ValueError(f"Unsupported space type for key {key}: {type(space)}")
                
        # Create a single Box space with the computed size
        return spaces.Box(low=0, high=1, shape=(total_size,), dtype=np.float32)

    @staticmethod
    def convert_dict_obs_to_single_obs(obs_dict, obs_space: spaces.Box, action_space: spaces.Box):
        # No need for dict_space argument

        single_space_obs = np.zeros(obs_space.shape[0], dtype=np.float32)
        index_counter = 0

        for agent_id, agent_obs in obs_dict.items():

            # Image:
            flat_values = agent_obs["image"].flatten()
            single_space_obs[index_counter:index_counter+len(flat_values)] = flat_values
            index_counter += len(flat_values)
            
            # Direction:
            one_hot_values = np.eye(4)[agent_obs["direction"]]  # 4 for Discrete(4)
            single_space_obs[index_counter:index_counter+4] = one_hot_values
            index_counter += 4

        return single_space_obs
    # def convert_dict_obs_to_single_obs(dict_obs: dict, dict_space: spaces.Dict) -> np.ndarray:
    #     single_obs_list = []
        
    #     for key, space in dict_space.spaces.items():
    #         # If the observation space is discrete, we'll use one-hot encoding.
    #         if isinstance(space, spaces.Discrete):
    #             one_hot = np.zeros(space.n, dtype=np.float32)
    #             one_hot[dict_obs[key]] = 1.0
    #             single_obs_list.append(one_hot)
            
    #         # If the observation space is a box, we'll flatten the observation.
    #         elif isinstance(space, spaces.Box):
    #             flattened_obs = dict_obs[key].astype(np.float32).reshape(-1)
    #             # If needed, you can normalize the observation to [0,1] range here.
    #             single_obs_list.append(flattened_obs)
            
    #         else:
    #             raise ValueError(f"Unsupported space type for key {key}: {type(space)}")
        
    #     # Concatenate all individual observations together to form a single observation vector.
    #     single_obs = np.concatenate(single_obs_list)
    #     return single_obs



    @staticmethod
    @nb.njit(cache=True)
    def one_hot(x: ndarray[np.int], dim_sizes: ndarray[np.int]) -> ndarray[np.uint8]:
        """
        Return a one-hot encoding of a 3D integer array,
        where each 2D slice is encoded separately.

        Parameters
        ----------
        x : ndarray[int] of shape (view_height, view_width, dim)
            3D array of integers to be one-hot encoded
        dim_sizes : ndarray[int] of shape (dim,)
            Number of possible values for each dimension

        Returns
        -------
        out : ndarray[uint8] of shape (view_height, view_width, sum(dim_sizes))
            One-hot encoding

        :meta private:
        """
        out = np.zeros((x.shape[0], x.shape[1], sum(dim_sizes)), dtype=np.uint8)

        dim_offset = 0
        for d in range(len(dim_sizes)):
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    k = dim_offset + x[i, j, d]
                    out[i, j, k] = 1

            dim_offset += dim_sizes[d]

        return out