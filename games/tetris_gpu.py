import datetime
import pathlib

import gymnasium as gym
import numpy as np
import torch
from tetris_gymnasium.wrappers.observation import RgbObservation


from .abstract_game import AbstractGame

from tetris_gymnasium.envs.tetris import Tetris


class MuZeroConfig:
    def __init__(self):
        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

        # Game
        self.observation_shape = (3, 24, 24)  # RGB observation of size 24x24
        self.action_space = list(range(8))  # Typical Tetris actions: left, right, rotate, down, drop, do nothing
        self.players = list(range(1))  # Single-player game

        # Self-Play
        self.num_workers = 4  # Utilize multiple CPU cores for self-play
        self.selfplay_on_gpu = True  # Use GPU for self-play to speed up the process
        self.max_moves = 10000  # Tetris games can be quite long
        self.num_simulations = 50  # Moderate number of simulations per move
        self.discount = 0.997  # High discount factor for long-term planning

        # Network
        self.network = "resnet"  # ResNet is good for image-based inputs
        self.support_size = 300  # Tetris can have high scores, increase support size
        self.downsample = "resnet"  # Downsample the input to extract features
        self.blocks = 6  # Moderate number of residual blocks
        self.channels = 64  # Increase channels for more expressive power

        # Training
        self.training_steps = 1000000  # Tetris is complex, require more training steps
        self.batch_size = 256  # Larger batch size for GPU utilization
        self.checkpoint_interval = 100  # Save model more frequently
        self.value_loss_weight = 0.25  # As recommended in the paper
        self.train_on_gpu = True  # Utilize GPU for training

        self.optimizer = "Adam"  # Adam often works well and requires less tuning
        self.weight_decay = 1e-4
        self.lr_init = 0.001  # Lower initial learning rate for stability
        self.lr_decay_rate = 0.95
        self.lr_decay_steps = 10000

        # Replay Buffer
        self.replay_buffer_size = 10000  # Larger replay buffer for more diverse experiences
        self.num_unroll_steps = 10
        self.td_steps = 50
        self.PER = True
        self.PER_alpha = 1.0  # Full prioritization as suggested in the paper

        # Reanalyze
        self.use_last_model_value = True
        self.reanalyse_on_gpu = True  # Utilize GPU for reanalysis

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 500e3:
            return 1.0
        elif trained_steps < 750e3:
            return 0.5
        else:
            return 0.25



class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = gym.make("tetris_gymnasium/Tetris", render_mode="human")
        self.env = RgbObservation(self.env)

        if seed is not None:
            # self.env.seed(seed)
            self.env.action_space.seed(seed)

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done, truncation, info = self.env.step(action)
        return np.transpose(observation, (2, 0, 1)), reward, done

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return list(range(7))

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        obs, _ = self.env.reset()
        return np.transpose(obs, (2, 0, 1))

    def close(self):
        """
        Properly close the game.
        """
        self.env.close()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        actions = {
            0: "Move left",
            1: "Move right",
            2: "Move down",
            3: "Rotate clockwise",
            4: "Rotate counterclockwise",
            5: "Hard drop",
            6: "Swap",
            7: "No op",
        }
        return f"{action_number}. {actions[action_number]}"
