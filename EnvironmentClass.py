from typing import Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

class GridWorldEnv(gym.Env):

    def __init__(self, size: int=5):
        super().__init__()

        self.size = size

        # Actions: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)

        self.agent_loc = np.array([-1,-1], dtype=np.int32)
        self.end_loc = np.array([-1,-1], dtype=np.int32)

        # Observation = agent position (x, y)
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size-1, shape=(2,), dtype=np.int32),
                "end": gym.spaces.Box(0, size-1, shape=(2,), dtype=np.int32),
            }
        )

        self.moves = gym.spaces.Discrete(4)

        # [rows,columns] so [y,x]
        self.map_actions_to_dir={
            0: np.array([0, 1]),  # right
            1: np.array([0, -1]),   # left
            2: np.array([1, 0]),  # down
            3: np.array([-1, 0])    # up
        }

    #returns dictionary with current positions of agent and goal
    def get_observations(self):
        """ Gathers current observations of the environment

        Returns:
            dictionary with current positions of agent and goal 
        """

        return {
            "agent": self.agent_loc.astype(np.int32),
            "end": self.end_loc.astype(np.int32)
        }
    
    def get_info(self):
        """Gathers additional information about the environment

        Returns:
            dictionary with distance from agent to goal
        """

        return {
            "distance_to_goal": np.linalg.norm(self.agent_loc - self.end_loc, ord=1)
        }
    
    def reset(self, seed: Optional[int]=None, options: Optional[dict]=None):
        """Resets environment and starts a new episode.
        
        Args:
            seed: Optional random seed for reproducibility.
            options: Optional dictionary of additional options for reset.
            
        Returns:
            A tuple containing the initial observation and info dictionary.
        """
        #used to seed RNG
        super().reset(seed=seed)

        #agent placed at random location in grid
        self.agent_loc = self.np_random.integers(0, self.size, size=2, dtype=np.int32)
        
        #make sure end position is not the same as agent position
        self.end_loc = self.agent_loc.copy()
        while np.array_equal(self.end_loc, self.agent_loc):
            self.end_loc = self.np_random.integers(
                0, self.size, size=2, dtype=np.int32
            )

        #gather information about the environment
        observation=self.get_observations()
        info=self.get_info()

        print("OBS:", observation)
        print("SPACE:", self.observation_space)
        print("CONTAINS?", self.observation_space.contains(observation))
        return observation, info
    
    def step(self, action):
        """Takes an action and updates the environment state.
        
        Args:
            action: An integer representing the action to take (0=up, 1=down, 2=left, 3=right).
        
        Returns:
            tuple containing new observation, reward, end, truncated, and info.
        """

        dir=self.map_actions_to_dir[action]

        self.agent_loc = np.clip(self.agent_loc + dir, 0, self.size-1)  

        end=np.array_equal(self.agent_loc, self.end_loc)

        truncated=False

        #punish agent more for being farther from the goal
        #reward of 1 for reaching the goal
        distance_to_goal=np.linalg.norm(self.agent_loc - self.end_loc, ord=1)
        reward= 1 if end else -0.1*distance_to_goal

        observation=self.get_observations()
        info=self.get_info()

        return observation, reward, end, truncated, info
    

gym.register(
    id="GridWorld-v0",
    entry_point="EnvironmentClass:GridWorldEnv",
)


    
