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
        # General
        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None # Use a single GPU for faster processing

        # Game
        self.observation_shape = (3, 24, 34)  # Typical Tetris board is 20x10, using 1 channel for piece positions
        self.action_space = list(range(8))  # 7 actions: move left, move right, rotate clockwise, rotate counterclockwise, soft drop, hard drop, hold
        self.players = list(range(1))  # Single-player game
        self.stacked_observations = 4  # Stack last 4 frames to capture piece movement

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = None  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        # Self-Play
        self.num_workers = 8  # Utilize multiple CPU cores for self-play
        self.selfplay_on_gpu = True  # Use GPU for self-play to speed up the process
        self.max_moves = 27000  # Limit very long games
        self.num_simulations = 50  # Moderate number of simulations per move
        self.discount = 0.997  # High discount factor for long-term planning
        self.temperature_threshold = None  # Use visit_softmax_temperature_fn throughout the game

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # Network
        self.network = "resnet"  # ResNet works well for grid-based games like Tetris
        self.support_size = 300  # Large support size for potentially high scores
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 6  # Moderate number of residual blocks
        self.channels = 128  # Increase channels for more expressive power
        self.reduced_channels_reward = 64
        self.reduced_channels_value = 64
        self.reduced_channels_policy = 64
        self.resnet_fc_reward_layers = [64, 64]
        self.resnet_fc_value_layers = [64, 64]
        self.resnet_fc_policy_layers = [64, 64]

        # Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / "tetris" / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        self.save_model = True
        self.training_steps = int(1e6)  # 1 million training steps
        self.batch_size = 1024  # Smaller batch size for more frequent updates
        self.checkpoint_interval = int(1e3)
        self.value_loss_weight = 0.25  # As recommended in the paper
        self.train_on_gpu = torch.cuda.is_available()

        self.optimizer = "Adam"  # Adam often works well and requires less tuning
        self.weight_decay = 1e-4
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.001  # Lower initial learning rate for stability
        self.lr_decay_rate = 0.95
        self.lr_decay_steps = 10000

        # Replay Buffer
        self.replay_buffer_size = int(1e6)  # Large replay buffer for diverse experiences
        self.num_unroll_steps = 5  # Unroll for 5 steps in the future
        self.td_steps = 10  # Use 10-step return for value estimation
        self.PER = True  # Use Prioritized Experience Replay
        self.PER_alpha = 1.0  # Full prioritization as suggested in the paper

        # Reanalyze
        self.use_last_model_value = True
        self.reanalyse_on_gpu = True

        # Self-play vs training ratio
        self.self_play_delay = 0
        self.training_delay = 0
        self.ratio = None  # Disable fixed ratio, let self-play and training run independently


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
