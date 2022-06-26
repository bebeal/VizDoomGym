import itertools
import warnings
from os import path
from collections import deque
from typing import Union, Tuple
import numpy as np
import gym
from gym import spaces
import vizdoom.vizdoom as vzd
import utils
from utils import LazyFrames
import torch
import cv2
import pygame

POSITION_BUFFER = [vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y,
                   vzd.GameVariable.POSITION_Z, vzd.GameVariable.ANGLE]

HEALTH_BUFFER = [vzd.GameVariable.HEALTH, vzd.GameVariable.ARMOR]

AMMO_BUFFER = [vzd.GameVariable.AMMO0,
               vzd.GameVariable.AMMO1,
               vzd.GameVariable.AMMO2,
               vzd.GameVariable.AMMO3,
               vzd.GameVariable.AMMO4,
               vzd.GameVariable.AMMO5,
               vzd.GameVariable.AMMO6,
               vzd.GameVariable.AMMO7,
               vzd.GameVariable.AMMO8,
               vzd.GameVariable.AMMO9]

# A fixed set of colors for each potential label
# for rendering an image.
# 256 is not nearly enough for all IDs, but we limit
# ourselves here to avoid hogging too much memory.
LABEL_COLORS = np.random.default_rng(42).uniform(25, 256, size=(256, 3)).astype(np.uint8)


class DoomEnv(gym.Env):
    def __init__(self,
                 scenarios: Union[str, Tuple[str, ...]],
                 assets: Union[str, Tuple[str, ...]] = None,
                 ini: str = None,
                 no_single_channel=False,
                 frame_skip: int = 1,
                 frame_stack: int = 1,
                 down_sample: Union[int, Tuple[int], Tuple[int, int]] = (None, None),
                 to_torch: bool = True,
                 normalize: bool = False,
                 max_buttons_pressed: int = 0,
                 encode_action: bool = True,
                 add_depth_buffer: bool = False,
                 add_labels_buffer: bool = False,
                 add_automap_buffer: bool = False,
                 add_audio_buffer: bool = False,
                 add_position_buffer: bool = False,
                 add_health_buffer: bool = False,
                 add_ammo_buffer: bool = False,
                 add_info_vars: Union[vzd.GameVariable, Tuple[vzd.GameVariable, ...]] = (),
                 shuffle_scenarios: float = 0.,
                 shuffle_assets: float = 0.,
                 shuffle_difficulty: float = 0.,
                 render_buffers: Tuple[int, ...] = None
                 ):
        """
        Highly Customizable Gym interface for ViZDoom.

        :param scenarios:           Scenario files
        :param assets:              Wad assets. Default: None.
                                    By default, uses "freedoom2.wad" assets
        :param ini:                 .ini settings (engine settings, including key bindings, etc). Default None.
                                    If None: by default VizDoom will use and create `_vizdoom.ini` in your working
                                    directory (if it does not exist).
        :param no_single_channel:   Flattens images with 1 color channel to be (H, W). Default: True. This allows for
                                    one to use frame_stack as a theoretical color channel and make the images returned
                                    (frame_stack, H, W).
        :param frame_skip:          Number of frames to skip per call to `step()`. Default 1.
        :param frame_stack:         Number of frames stacked and returned in observation when `step()` is called.
                                    Default 1.
                                    Discards the single oldest frame, and appends on a single fresh frame per call to
                                    `step()`
        :param down_sample:         Resolution of image buffer, overrides value set in scenario file.
                                    Default: (None, None).
                                    If None for either value or both, uses the values specified in the scenario file.
        :param to_torch:            Whether to change the numpy observations to torch tensors. Default: True.
        :param normalize:           Whether to cast image buffers to float32 and divide values by 255. Default: True.
        :param max_buttons_pressed: Defines the number of binary buttons that can be selected at once. Default: 1.
                                    Only used if encode_action, otheriwise ignored. Should be >= 0.
                                    If < 0 a RuntimeError is raised.
                                    If == 0, the binary action space becomes MultiDiscrete([2] * num_binary_buttons)
                                    and [0, num_binary_buttons] number of binary buttons can be selected.
                                    If > 0, the binary action space becomes Discrete(2**n)
                                    and [0, max_buttons_pressed] number of binary buttons can be selected.
        :param encode_action:       Determines self.action_space, dependent on which available_game_actions() are
                                    specified by the scenario file, and what valid actions that can be sent to
                                    `step(action)`.
                                    Default: False. (Not used by default).
                                    If True:
                                        Action space can be a single one of binary/continuous action space, or a Dict
                                        containing both.
                                        "binary":
                                            if max_buttons_pressed == 0: MultiDiscrete([2] * num_binary_buttons)
                                            if max_buttons_pressed > 1: Discrete(n) where n is the number of environment actions that have
                                                                        0 <= max_buttons_pressed bits set
                                        "continuous":
                                            Box(-max_value, max_value, (num_delta_buttons,), np.float32)
                                    else:
                                        Action space is Box(-np.inf, np.inf, ({n},), np.float32) where {n} is defined
                                        to be the number of available_game_buttons as specified by the scenario file.
                                        And the action sent to `step()` is directly sent to the VizDoom environment.
        :param add_depth_buffer:    Whether to add the depth_buffer to the observation_space. Default: False.
                                    If True: overrides value set in scenario file,
                                    else: uses setting specified in scenario file.
        :param add_labels_buffer:   Whether to add the labels_buffer to the observation_space. Default: False.
                                    If True: overrides value set in scenario file,
                                    else: uses setting specified in scenario file.
        :param add_automap_buffer:  Whether to add the automap_buffer to the observation_space. Default: False.
                                    If True: overrides value set in scenario file,
                                    else: uses setting specified in scenario file.
        :param add_audio_buffer:    Whether to add the audio_buffer to the observation_space. Default: False.
                                    If True: overrides value set in scenario file,
                                    else: uses setting specified in scenario file.
        :param add_position_buffer: Whether to add the POSITION_BUFFER to the observation_space. Default: False.
                                    If True: add them as game_variables
                                    else: uses game_variables specified in scenario file.
        :param add_health_buffer:   Whether to add the HEALTH_BUFFER to the observation_space. Default: False.
                                    If True: add them as game_variables
                                    else: uses game_variables specified in scenario file.
        :param add_ammo_buffer:     Whether to add the AMMO_BUFFER to the observation_space. Default: False.
                                    If True: add them as game_variables
                                    else: uses game_variables specified in scenario file.
        :param add_info_vars:       Enables and adds these game_variables to the game state, passed back via `info` in
                                    `step()`. Default: ().
                                    Unless these variables are already specified in the scenario file this will not add
                                    these variables to the observation, only the info.
        :param shuffle_scenarios:   Probability to randomize the scenario per call to `reset()`. Default: 0.
                                    (0 probability = no shuffling, 1 = shuffle every call to `reset()`)
        :param shuffle_assets:      Probability to randomize the assets per call to `reset()`. Default: 0.
                                    (0 probability = no shuffling, 1 = shuffle every call to `reset()`)
        :param shuffle_difficulty:  Probability to randomize the doom_skill difficulty per call to `reset()`.
                                    Default: 0.
                                    (0 probability = no shuffling, 1 = shuffle every call to `reset()`)
        :param render_buffers:      Determines which buffers to render when calling `env.render()`. Default None.
                                    If None, renders all activated buffers.
                                    Otherwise, tuple containing 0/1 indicates whether or not to render the buffer at
                                    that index. Ex: If Depth, Automap, and Audio buffer are enabled you can send a tuple
                                    specifying which ones should be rendered, i.e: (1, 0, 1) would result in the depth
                                    buffer, and audio buffer being rendered.
                                    As a special case for rendering the audio buffer:
                                    1: line wave option
                                    2: circular wave option
                                    3: filled circular wave option
        """
        super(DoomEnv, self).__init__()
        if isinstance(scenarios, str):
            scenarios = (scenarios, )
        if isinstance(assets, str):
            assets = (assets, )
        if isinstance(down_sample, int):
            down_sample = (down_sample, down_sample)
        if len(down_sample) == 1:
            down_sample = (down_sample[0], down_sample[0])
        if not isinstance(add_info_vars, tuple):
            add_info_vars = (add_info_vars,)

        if len(scenarios) == 0:
            raise RuntimeError(f"Must specify atleast 1 scenario")
        if frame_skip <= 0:
            warnings.warn(f"frame_skip = {frame_skip} <= 0.\nSetting to 1")
            frame_skip = 1
        if frame_stack <= 0:
            warnings.warn(f"frame_stack = {frame_stack} <= 0.\nSetting to 1")
            frame_stack = 1

        # save configuration
        self.scenarios = scenarios
        self.assets = assets
        self.ini = ini
        self.no_single_channel = no_single_channel
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.down_sample = down_sample
        self.to_torch = to_torch
        self.normalize = normalize
        self.max_buttons_pressed = max_buttons_pressed
        self.encode_action = encode_action
        self.shuffle_assets = shuffle_assets
        self.shuffle_scenarios = shuffle_scenarios
        self.shuffle_difficulty = shuffle_difficulty
        self.render_buffers = render_buffers

        # configure vizdoom instance
        self.__set_game()
        self.__load_scenario()
        self.__set_default_configs()
        if assets is not None:
            asset_idx = None
            if len(assets) > 0:
                asset_idx = 0
            self.__load_asset(asset_idx)
        self.__load_ini(ini)
        if down_sample[0] is None:
            down_sample = (self.game.get_screen_height(), down_sample[1])
        if down_sample[1] is None:
            down_sample = (down_sample[0], self.game.get_screen_width())
        self.down_sample = down_sample
        self.__set_buffers(add_depth_buffer, add_labels_buffer, add_automap_buffer, add_audio_buffer,
                           add_position_buffer, add_health_buffer, add_ammo_buffer, add_info_vars)
        self.__set_observation_space()
        self.__set_action_space()
        self.__init_game()

    def reset(self, *, seed=None, return_info=False, options=None):
        if self.shuffle_scenarios or self.shuffle_assets or self.shuffle_difficulty:
            # with probability shuffle_scenarios, change the scenario
            shuffle_scenario = self.shuffle_scenarios and torch.rand(1)[0].item() < self.shuffle_scenarios
            self.__reset_game(randomize=shuffle_scenario)
            # with probability shuffle_assets, change the asset
            if self.shuffle_assets and torch.rand(1)[0].item() < self.shuffle_assets:
                self.__randomize_asset()
            # with probability shuffle_difficulty, change the difficulty
            if self.shuffle_difficulty and torch.rand(1)[0].item() < self.shuffle_difficulty:
                self.__randomize_difficulty()

            # have to reconfigure vizdoom instance for changes to take affect
            self.__init_game()
        else:
            # no reconfiguration needed
            self.game.new_episode()

        if seed is not None:
            # set seed
            self.game.set_seed(seed)

        for i in range(self.frame_stack):
            self.__append_new_frame()

        observation = self.__get_stacked_observation_frames()
        if not return_info:
            return observation
        else:
            return observation, self.__get_info()

    def step(self, action):
        info = self.__get_info()
        if self.encode_action:
            assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
            action = self.__build_env_action(action)
        reward = self.game.make_action(action, self.frame_skip)
        done = int(self.game.is_episode_finished())
        self.__append_new_frame()
        observation = self.__get_stacked_observation_frames()
        return observation, reward, done, info

    def render(self, mode="human"):
        render_image = self.__build_human_render_image()
        if mode == "rgb_array":
            return render_image
        elif mode == "human":
            render_image = render_image.transpose(1, 0, 2)  # HWC -> WHC
            if self.window_surface is None:
                pygame.init()
                pygame.display.set_caption("ViZDoom")
                self.window_surface = pygame.display.set_mode(render_image.shape[:2])

            surf = pygame.surfarray.make_surface(render_image)
            self.window_surface.blit(surf, (0, 0))
            pygame.display.update()
        else:
            return self.is_open

    def close(self):
        if self.window_surface:
            pygame.quit()
            self.is_open = False
        if self.has_audio_buffer:
            buffer_idx = 1 + self.has_depth_buffer + self.has_labels_buffer + self.has_audio_buffer
            if self.render_buffers[buffer_idx]:
                self.audio_visualizer.__exit__()

    def serialize(self):
        """
        Returns a dictionary containing environment information.
        """
        return {"observation_space": self.observation_space.__repr__(), "action_space": self.action_space.__repr__(),
                "scenarios": self.scenarios, "assets":self.assets, "ini":self.ini,
                "no_single_channel":self.no_single_channel, "frame_skip": self.frame_skip,
                "frame_stack": self.frame_stack, "down_sample": self.down_sample, "to_torch": self.to_torch,
                "normalize": self.normalize, "max_buttons_pressed": self.max_buttons_pressed,
                "encode_action": self.encode_action, "has_depth_buffer": self.has_depth_buffer,
                "has_labels_buffer": self.has_labels_buffer, "has_automap_buffer": self.has_automap_buffer,
                "has_audio_buffer": self.has_audio_buffer, "add_position_buffer": self.num_position_buffer,
                "add_health_buffer": self.num_health_buffer, "add_ammo_buffer": self.num_ammo_buffer,
                "info_vars": self.info_vars, "shuffle_assets": self.shuffle_assets,
                "shuffle_scenarios": self.shuffle_scenarios, "shuffle_difficulty": self.shuffle_difficulty,
                "render_buffers": self.render_buffers}

    """
    Lower level functions beneath this point
    """

    def __exit__(self):
        self.close()

    def __set_game(self):
        """
        Resets vizdoom game.
        """
        self.game = vzd.DoomGame()

    def __reset_game(self, randomize=False):
        """
        Resets and configures vizdoom instance.
        """
        self.close()
        self.__set_game()
        if randomize:
            self.__randomize_scenario()
        else:
            self.__load_scenario()
            self.__set_default_configs()
        self.__load_ini(self.ini)
        self.__reset_buffers()
        self.__set_observation_space()
        self.__set_action_space()

    def __load_scenario(self, scenario_idx=0):
        """
        Set the games scenario.
        scenario_idx = None: Don't try and set scenario (i.e. keep current scenario loaded).
        scenario_idx < 0: randomize scenario from list of scenario.
        scenario_idx >= 0: set the scenario at this index if path exists, else crash.
        """
        if scenario_idx is not None:
            if scenario_idx < 0:
                scenario_idx = torch.randint(low=0, high=len(self.scenarios), size=(1,))[0].item()
            scenario_name = self.scenarios[scenario_idx]
            if "/" not in scenario_name:
                scenario_name = path.join("./scenarios", scenario_name)
            if path.exists(scenario_name):
                self.scenario_name = scenario_name
                self.scenario_idx = scenario_idx
                self.game.load_config(scenario_name)
            else:
                # error instead of warning as no fallback scenario
                raise RuntimeError(f"Can't find scenario at {scenario_name}")

    def __set_default_configs(self):
        """
        Default configuration for using Gym-like interface (`step()`, `render()`, etc).
        """
        self.game.set_window_visible(False)
        self.window_surface = None
        self.is_open = True

    def __load_asset(self, asset_idx=0):
        """
        Set the games asset.
        asset_idx = None: Don't try and set asset (i.e. keep current asset loaded).
        asset_idx < 0: randomize asset from list of assets.
        asset_idx >= 0: set the asset at this index if path exists.
        """
        if asset_idx is not None and self.assets is not None:
            if asset_idx < 0:
                asset_path = self.assets[torch.randint(low=0, high=len(self.assets), size=(1,))[0].item()]
            else:
                asset_path = self.assets[asset_idx]
            if path.exists(asset_path):
                self.game.set_doom_game_path(asset_path)
            else:
                warnings.warn(f"Can't find asset at {asset_path}. Keeping previous asset.")

    def __load_ini(self, ini):
        """
        Load given .ini settings (engine settings, including key bindings, etc).
        If None: by default VizDoom will use and create `_vizdoom.ini` in your working directory (if it does not exist).
        """
        if ini is not None:
            if path.exists(ini):
                self.game.set_doom_config_path(ini)
            else:
                warnings.warn(f"Can't find .ini at {ini}")

    def __set_buffers(self, add_depth_buffer, add_labels_buffer, add_automap_buffer, add_audio_buffer,
                      add_position_buffer, add_health_buffer, add_ammo_buffer, add_info_vars):
        """
        Determine which buffers/game_variables to enable or are already enabled via the scenario file
        """
        self.has_depth_buffer = add_depth_buffer or self.game.is_depth_buffer_enabled()
        self.has_labels_buffer = add_labels_buffer or self.game.is_labels_buffer_enabled()
        self.has_automap_buffer = add_automap_buffer or self.game.is_automap_buffer_enabled()
        self.has_audio_buffer = add_audio_buffer or self.game.is_audio_buffer_enabled()
        if add_position_buffer:
            self.__add_game_variables(POSITION_BUFFER)
        if add_health_buffer:
            self.__add_game_variables(HEALTH_BUFFER)
        if add_ammo_buffer:
            self.__add_game_variables(AMMO_BUFFER)
        if len(add_info_vars) != 0:
            self.__add_game_variables(add_info_vars)
        # determine index of game variables
        self.num_position_buffer, self.num_health_buffer, self.num_ammo_buffer, self.num_game_buffer, self.num_info_vars = self.__parse_game_variables(add_info_vars)
        self.info_vars = add_info_vars
        if self.render_buffers is None:
            self.render_buffers = (1,) * (1 + self.has_depth_buffer + self.has_labels_buffer + self.has_automap_buffer +
                                     self.has_audio_buffer + (self.num_position_buffer > 0) +
                                     (self.num_health_buffer > 0) + (self.num_ammo_buffer > 0) +
                                     (self.num_game_buffer > 0))

    def __reset_buffers(self):
        """
        Set state of buffers to initial state set when `self.__set_buffers()` was called via the constructor.
        """
        self.__set_buffers(self.has_depth_buffer, self.has_labels_buffer, self.has_automap_buffer,
                           self.has_audio_buffer, self.num_position_buffer, self.num_health_buffer,
                           self.num_ammo_buffer, self.info_vars)

    def __init_game(self):
        """
        Re-initializes vizdoom instance.
        """
        try:
            self.game.init()
        except Exception as e:
            print("Audio fix")
            # Audio fix https://github.com/mwydmuch/ViZDoom/pull/486#issuecomment-895128340
            self.game.add_game_args("+snd_efx 0")
            self.game.init()

    def __build_human_render_image(self):
        """
        Stack all available buffers into one for human consumption.
        """
        game_state = self.game.get_state()
        valid_buffers = game_state is not None

        if not valid_buffers:
            # Return a blank image
            num_enabled_buffers = np.array(list(self.render_buffers)).sum()
            # always 3 color channels regardless of screen type
            img = np.zeros(
                (self.game.get_screen_height(), self.game.get_screen_width() * num_enabled_buffers, 3,), dtype=np.uint8
            )
            return img

        if self.render_buffers[0]:
            image_list = [self.__force_3_channels(game_state.screen_buffer)]

        buffer_idx = 1
        if self.has_depth_buffer:
            buffer_idx += 1
            if self.render_buffers[buffer_idx - 1]:
                image_list.append(self.__force_3_channels(game_state.depth_buffer))

        if self.has_labels_buffer:
            buffer_idx += 1
            if self.render_buffers[buffer_idx - 1]:
                # Give each label a fixed color.
                # We need to connect each pixel in labels_buffer to the corresponding
                # id via `value``
                labels_rgb = np.zeros((self.down_sample[0], self.down_sample[1], 3))
                labels_buffer = game_state.labels_buffer
                for label in game_state.labels:
                    color = LABEL_COLORS[label.object_id % 256]
                    labels_rgb[labels_buffer == label.value] = color
                image_list.append(labels_rgb)

        if self.has_automap_buffer:
            buffer_idx += 1
            if self.render_buffers[buffer_idx - 1]:
                image_list.append(self.__force_3_channels(game_state.automap_buffer))

        if self.has_audio_buffer:
            buffer_idx += 1
            if self.render_buffers[buffer_idx - 1]:
                surf = self.audio_visualizer.step(game_state.audio_buffer)
                image_list.append(surf)

        return np.concatenate(image_list, axis=1)

    def __force_3_channels(self, buffer):
        """
        Forces buffer to be (H, W, 3) and in RGB order so that the buffer can be rendered
        """
        if len(buffer.shape) > 2:
            if self.screen_type == 0:
                buffer = buffer.transpose(1, 2, 0)
            elif self.screen_type == 3:
                buffer = buffer[:, :, :3]
            elif self.screen_type == -3:
                buffer = buffer[:, :, 1:]
            if self.rearrange_rgb:
                buffer = buffer[:, :, ::-1]
        else:
            buffer = np.repeat(np.expand_dims(buffer, axis=self.lift_axis), repeats=3, axis=self.lift_axis)
        return buffer

    def __parse_binary_buttons(self, env_action, agent_action):
        """
        Only used if encode_action.
        Encodes the binary agent action into environment action.
        No binary buttons being defined (num_binary_buttons==0) results in a noop.
        """
        if self.num_binary_buttons != 0:
            if self.num_delta_buttons != 0:
                agent_action = agent_action["binary"]

            if isinstance(agent_action, int):
                agent_action = self.button_map[agent_action]

            if len(agent_action) != 0:
                # binary actions offset by number of delta buttons
                env_action[self.num_delta_buttons:] = agent_action

    def __parse_delta_buttons(self, env_action, agent_action):
        """
        Only used if encode_action.
        Encodes the continuous agent action into environment action.
        No delta buttons being defined (num_delta_buttons==0) results in a noop.
        All n Box()'s in the continuous action space map directly to the first n values of the environment action.
        """
        if self.num_delta_buttons != 0:
            if self.num_binary_buttons != 0:
                agent_action = agent_action["continuous"]

            env_action[0:self.num_delta_buttons] = agent_action

    def __build_env_action(self, agent_action):
        """
        Only used if encode_action.
        Encodes the given agent_action to the environment_action that is sent to the vizdoom instance.
        """
        # encode users action as environment action
        env_action = np.array([0 for _ in range(self.num_delta_buttons + self.num_binary_buttons)], dtype=np.float32)
        self.__parse_delta_buttons(env_action, agent_action)
        self.__parse_binary_buttons(env_action, agent_action)
        return env_action

    def __parse_available_buttons(self):
        """
        Only used if encode_action.
        Parses the currently available game buttons.
        reorganizes all delta buttons to be prior to any binary buttons.
        returns list of delta buttons, number of delta buttons, number of binary buttons.
        list of delta buttons contains Box type for each delta button.
        """
        delta_buttons = []
        binary_buttons = []
        for button in self.game.get_available_buttons():
            if vzd.is_delta_button(button) and button not in delta_buttons:
                delta_buttons.append(button)
            else:
                binary_buttons.append(button)
        # force all delta buttons to be first before any binary buttons
        self.game.set_available_buttons(delta_buttons + binary_buttons)
        self.num_delta_buttons = len(delta_buttons)
        self.num_binary_buttons = len(binary_buttons)
        if self.num_binary_buttons == self.num_delta_buttons == 0:
            raise RuntimeError("No game buttons defined. Must specify game buttons using `available_buttons` in the "
                               "config file.")

        # check for valid max_buttons_pressed
        if self.max_buttons_pressed > self.num_binary_buttons > 0:
            warnings.warn(
                f"max_buttons_pressed={self.max_buttons_pressed} "
                f"> number of binary buttons defined={self.num_binary_buttons}. "
                f"Clipping max_buttons_pressed to {self.num_binary_buttons}.")
            self.max_buttons_pressed = self.num_binary_buttons
        elif self.max_buttons_pressed < 0:
            raise RuntimeError(f"max_buttons_pressed={self.max_buttons_pressed} < 0. Should be >= 0. ")

    def __get_binary_action_space(self):
        """
        Only used if encode_action.
        return binary action space: either (Discrete(n)|MultiDiscrete([2,]*num_binary_buttons)).
        """
        if self.max_buttons_pressed == 0:
            button_space = gym.spaces.MultiDiscrete([2,] * self.num_binary_buttons)
        else:
            self.button_map = [
                np.array(list(action)) for action in itertools.product((0, 1), repeat=self.num_binary_buttons)
                if (self.max_buttons_pressed >= sum(action) >= 0)
            ]
            button_space = gym.spaces.Discrete(len(self.button_map))
        return button_space

    def __get_continuous_action_space(self):
        """
        Only used if encode_action.
        return continuous action space: Box(-max_value, max_value, (num_delta_buttons,), np.float32).
        """
        return gym.spaces.Box(-np.inf, np.inf, (self.num_delta_buttons,), dtype=np.float32,)

    def __get_action_space(self):
        """
        returns action space of environment, dependent on self.encode_actions, self.max_buttons_pressed and which
        available_game_actions() are specified by the scenario file.
        if self.encode_action:
            Action space can be a single one of binary/continuous action space, or a Dict containing both.
                "binary":
                    if max_buttons_pressed == 0: MultiDiscrete([2] * num_binary_buttons)
                    if max_buttons_pressed > 1: Discrete(n) where n is the number of environment actions that have
                                                0 <= max_buttons_pressed bits set
                "continuous":
                    Box(-max_value, max_value, (num_delta_buttons,), np.float32)
        else:
            Action space is Box(-np.inf, np.inf, ({x},), np.float32) where {x} is defined
            to be the number of available_game_buttons() as specified by the scenario file.
            And the action sent to `step()` is directly sent to the VizDoom environment.
        """
        if self.encode_action:
            self.__parse_available_buttons()
            if self.num_delta_buttons == 0:
                return self.__get_binary_action_space()
            elif self.num_binary_buttons == 0:
                return self.__get_continuous_action_space()
            else:
                return gym.spaces.Dict({"binary": self.__get_binary_action_space(),
                                        "continuous": self.__get_continuous_action_space()})
        else:
            return spaces.Box(-np.inf, np.inf, (self.game.get_available_buttons_size(),))

    def __set_action_space(self):
        """
        Set self.action_space.
        """
        self.action_space = self.__get_action_space()

    def __parse_observation_space(self):
        """
        Return observation space as defined by current self.game configuration.
        """
        # channels = C, shape: CHW//HWC
        channels = self.channels
        if self.no_single_channel and self.channels == 1:
            channels = 0
        observation_space = [self.__get_screen_box(channels)]
        buffer_idx = 1
        if self.has_depth_buffer:
            buffer_idx += 1
            # channels = 1, shape: 1HW//HW1
            self.game.set_depth_buffer_enabled(True)
            observation_space.append(self.__get_screen_box(not self.no_single_channel))
        if self.has_labels_buffer:
            buffer_idx += 1
            # channels = 1, shape: 1HW//HW1
            self.game.set_labels_buffer_enabled(True)
            observation_space.append(self.__get_screen_box(not self.no_single_channel))
        if self.has_automap_buffer:
            buffer_idx += 1
            # channels = C, shape: CHW//HWC
            self.game.set_automap_buffer_enabled(True)
            self.game.set_automap_mode(vzd.AutomapMode.OBJECTS)
            self.game.set_automap_rotate(False)
            observation_space.append(self.__get_screen_box(channels))
        if self.has_audio_buffer:
            buffer_idx += 1
            # C = 1260 * audio buffer size, shape: C2
            self.game.set_audio_buffer_enabled(True)
            self.game.set_audio_sampling_rate(vzd.SamplingRate.SR_44100)
            self.game.set_audio_buffer_size(4 if self.frame_skip <= 4 else self.frame_skip)
            if self.render_buffers[buffer_idx - 1] >= 2:
                fill = False
                if self.render_buffers[buffer_idx - 1] == 3:
                    fill = True
                self.audio_visualizer = utils.DirectionalAudioVisualizer(self.game.get_audio_sampling_rate(), self.game.get_audio_buffer_size(), self.down_sample[0], self.down_sample[1], fill=fill)
            else:
                self.audio_visualizer = utils.FrequencyAudioVisualizer(self.game.get_audio_sampling_rate(), self.game.get_audio_buffer_size(), self.down_sample[0], self.down_sample[1])
            observation_space.append(gym.spaces.Box(-1, 1, (self.frame_stack, 1260 * self.game.get_audio_buffer_size(), 2), dtype=np.float32 if self.to_torch else np.int16))
        if self.num_position_buffer > 0:
            observation_space.append(gym.spaces.Box(-np.inf, np.inf, (self.frame_stack, self.num_position_buffer), dtype=np.float32))
        if self.num_health_buffer > 0:
            observation_space.append(gym.spaces.Box(-np.inf, np.inf, (self.frame_stack, self.num_health_buffer), dtype=np.float32))
        if self.num_ammo_buffer > 0:
            observation_space.append(gym.spaces.Box(-np.inf, np.inf, (self.frame_stack, self.num_ammo_buffer), dtype=np.float32))
        if self.num_game_buffer > 0:
            observation_space.append(gym.spaces.Box(-np.inf, np.inf, (self.frame_stack, self.num_game_buffer), dtype=np.float32))

        if len(observation_space) == 1:
            return observation_space[0]
        else:
            return gym.spaces.Tuple(observation_space)

    def __get_screen_box(self, channels=0):
        """
        General use util function.
        """
        channels_first = self.lift_axis == 0
        if self.to_torch:
            channels_first = True
        return utils.get_box(self.frame_stack, self.down_sample[0], self.down_sample[1], channels, channels_first, self.raw_type)

    def __parse_screen_type(self):
        """
        General use util function.
        Determines which 'screen type' we have, 3 possible options indicated by SCREEN_FORMATS
        0 == 3HW np.uint8
        1 == HW1 np.uint32
        2 == HW3 np.int32
        3 == HW4 np.int8
        used to determine shape for screen, depth, labels, automap buffer
        """
        screen_format = self.game.get_screen_format()
        screen_type = 0
        channels = 3
        raw_type = np.uint8
        if screen_format in [vzd.ScreenFormat.RGBA32, vzd.ScreenFormat.BGRA32, vzd.ScreenFormat.ARGB32, vzd.ScreenFormat.ABGR32]:
            screen_type = 3
            channels = 4
            if screen_format in [vzd.ScreenFormat.ARGB32, vzd.ScreenFormat.ABGR32]:
                screen_type = -3
        elif screen_format in [vzd.ScreenFormat.RGB24, vzd.ScreenFormat.BGR24]:
            screen_type = 2
            channels = 3
        elif screen_format in [vzd.ScreenFormat.GRAY8, vzd.ScreenFormat.DOOM_256_COLORS8]:
            screen_type = 1
            channels = 1

        lift_axis = -1
        if screen_format in [vzd.ScreenFormat.CRCGCB, vzd.ScreenFormat.CBCGCR]:
            lift_axis = 0
        if self.normalize or self.to_torch:
            raw_type = np.float32
        self.rearrange_rgb = screen_format in [vzd.ScreenFormat.BGRA32, vzd.ScreenFormat.ABGR32, vzd.ScreenFormat.BGR24, vzd.ScreenFormat.CBCGCR]
        self.lift_axis = lift_axis
        self.raw_type = raw_type
        self.channels = channels
        self.screen_type = screen_type

    def __set_observation_space(self):
        """
        Sets the observation space
        """
        self.__parse_screen_type()
        self.observation_space = self.__parse_observation_space()
        self.frames = deque([], maxlen=self.frame_stack)
        if isinstance(self.observation_space, gym.spaces.Tuple):
            self.frames = [deque([], maxlen=self.frame_stack) for _ in range(len(self.observation_space))]

    def __get_stacked_observation_frames(self):
        """
        Get all the accumulated stacked frames. This is what is returned to the user per call to `step()`.
        Handles multiple buffers (i.e. observation space of Tuples) and in this case returns a tuple of stacked frames.
        """
        if isinstance(self.observation_space, gym.spaces.Tuple):
            obs = []
            num_image_frames = 1 + self.has_depth_buffer + self.has_labels_buffer + self.has_audio_buffer
            for i in range(len(self.observation_space)):
                obs.append(self.__transform_stacked_frame(np.array(LazyFrames(list(self.frames[i])))))
                num_image_frames -= 1
            return tuple(obs)
        else:
            return self.__transform_stacked_frame(np.array(LazyFrames(list(self.frames))))

    def __get_single_frame(self):
        """
        Gets the current single freshest frame/s.
        Handles multiple buffers (i.e. observation space of Tuples) and in this case returns a tuple of frames.
        """
        observation = []
        state = self.game.get_state()

        if state:
            screen_buffer = state.screen_buffer
            observation.append(self.__transform_single_frame(screen_buffer))
            if self.has_depth_buffer:
                observation.append(self.__transform_single_frame(state.depth_buffer))
            if self.has_labels_buffer:
                observation.append(self.__transform_single_frame(state.labels_buffer))
            if self.has_automap_buffer:
                observation.append(self.__transform_single_frame(state.automap_buffer))
            if self.has_audio_buffer:
                observation.append(self.__transform_single_frame(state.audio_buffer, False))
            if self.num_position_buffer:
                observation.append(self.__transform_single_frame(np.array(state.game_variables[self.position_idxs]), False))
            if self.num_health_buffer:
                observation.append(self.__transform_single_frame(np.array(state.game_variables[self.health_idxs]), False))
            if self.num_ammo_buffer:
                observation.append(self.__transform_single_frame(np.array(state.game_variables[self.ammo_idxs]), False))
            if self.num_game_buffer:
                observation.append(self.__transform_single_frame(np.array(state.game_variables[self.game_var_idxs]), False))
        else:
            obs_space = self.observation_space
            # there is no state in the terminal step, so a "zero observation is returned instead"
            if not isinstance(self.observation_space, gym.spaces.Tuple):
                obs_space = [obs_space]

            for space in obs_space:
                observation.append(self.__transform_single_frame(np.zeros(space.shape[1:], dtype=space.dtype), False))

        observation = tuple(observation)
        if len(observation) == 1:
            observation = observation[0]
        return observation

    def __append_new_frame(self):
        """
        Appends most recent frame to frame buffer/s.
        Handles multiple buffers (i.e. observation space of Tuples).
        """
        if isinstance(self.observation_space, gym.spaces.Tuple):
            observation = self.__get_single_frame()
            #self.frames[:] = observation
            for i in range(len(self.observation_space)):
                self.frames[i].append(observation[i])
        else:
            self.frames.append(self.__get_single_frame())

    def __transform_stacked_frame(self, observation):
        """
        Apply any final transformations to stacked observation
        """
        observation = self.__to_torch(observation, False)
        return observation

    def __transform_single_frame(self, observation, image_frame=True):
        """
        Down Samples if necessary and lifts dimensions of observations with no channels to have a channel from
        HW->1HW//HW1 depending on what axis the channels are according to the screen type and normalizes if necessary.
        """
        if image_frame:
            observation = self.__downscale(observation)
            observation = self.__lift(observation)
            observation = self.__normalize(observation)
        observation = self.__to_torch(observation, image_frame)
        return observation

    def __downscale(self, observation):
        """
        downscale observation
        """
        if self.down_sample != (self.game.get_screen_height(), self.game.get_screen_width()):
            if self.screen_type == 0 and len(observation) > 2:
                observation = observation.transpose(1, 2, 0)  # CHW -> HWC
            observation = cv2.resize(observation, (self.down_sample[1], self.down_sample[0]), interpolation=cv2.INTER_LINEAR)
            if self.screen_type == 0 and len(observation) > 2:
                observation = observation.transpose(2, 0, 1)  # HWC -> CHW
        return observation

    def __lift(self, observation):
        """
        Add single color channel to observation
        """
        if len(observation.shape) == 2 and not self.no_single_channel:
            # HW -> (HW1|1HW)
            observation = np.expand_dims(observation, axis=self.lift_axis)
        return observation

    def __normalize(self, observation):
        """
        normalize observation by making float and dividing by 255
        """
        if self.normalize:
            observation = observation.astype(dtype=np.float32) / 255.
        return observation

    def __to_torch(self, observation, image_frame=True):
        """
        Converts observation from NumPy array to PyTorch Tensor
        and flip the color channels to be before H,W
        """
        if self.to_torch:
            observation = observation.astype(dtype=np.float32)
            if image_frame and self.lift_axis != 0 and len(observation.shape) > 2:
                observation = observation.transpose(2, 0, 1)
            observation = torch.from_numpy(observation)
        return observation

    def __parse_game_variables(self, add_info_vars):
        """
        parses game variables to determine index of POSITIONS/HEALTH/AMMO, any other game_vars and add_info_vars vars
        incase the scenario config file already defined them.
        """
        game_variables = self.game.get_available_game_variables()

        self.position_idxs = []
        self.health_idxs = []
        self.ammo_idxs = []
        self.game_var_idxs = []
        self.info_idxs = []

        for i in range(len(game_variables)):
            game_var = game_variables[i]
            if game_var in POSITION_BUFFER:
                self.position_idxs.append(i)
            elif game_var in HEALTH_BUFFER:
                self.health_idxs.append(i)
            elif game_var in AMMO_BUFFER:
                self.ammo_idxs.append(i)
            else:
                self.game_var_idxs.append(i)
            # can be in both POSITIONS/HEALTH/AMMO and info section
            if game_var in add_info_vars:
                self.info_idxs.append(i)
        return len(self.position_idxs), len(self.health_idxs), len(self.ammo_idxs), len(self.game_var_idxs), len(self.info_idxs)

    def __add_game_variables(self, variables):
        """
        adds variables to available game variables
        """
        for i in range(len(variables)):
            self.game.add_available_game_variable(variables[i])

    def __get_info(self):
        """
        Get the info dictionary containing value of game variables given by self.info_vars.
        As long as these vars were not already pre-specified in the scenario file, these vars are not included in the
        observation, only in info.
        """
        if self.num_info_vars:
            state = self.game.get_state()
            info = dict()
            for i in range(self.num_info_vars):
                info[str(self.info_vars[i])] = np.array([state.game_variables[self.info_idxs[i]]])
            return info
        else:
            return {}

    def __randomize_scenario(self):
        """
        Randomize current specified scenario.
        """
        self.__load_scenario(-1)
        self.__set_default_configs()

    def __randomize_asset(self):
        """
        Randomize current specified asset.
        """
        self.__load_asset(-1)

    def __randomize_difficulty(self):
        """
        Randomize current specified doom_skill i.e. difficulty.
        doom_skill range: [0, 5].
        """
        self.game.set_doom_skill(torch.randint(low=0, high=5, size=(1,))[0].item())
