from os import path
from collections import deque
import numpy as np
import gym
from gym import spaces
import vizdoom.vizdoom as vzd
from utils import LazyFrames
import torch
import cv2
import pygame

BASE_PATH = "scenarios"

POSITIONS = [vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y,
             vzd.GameVariable.POSITION_Z, vzd.GameVariable.ANGLE]

HEALTH = [vzd.GameVariable.HEALTH, vzd.GameVariable.ARMOR]

AMMO = [vzd.GameVariable.AMMO0,
        vzd.GameVariable.AMMO1,
        vzd.GameVariable.AMMO2,
        vzd.GameVariable.AMMO3,
        vzd.GameVariable.AMMO4,
        vzd.GameVariable.AMMO5,
        vzd.GameVariable.AMMO6,
        vzd.GameVariable.AMMO7,
        vzd.GameVariable.AMMO8,
        vzd.GameVariable.AMMO9]

SCREEN_FORMATS = [
    [vzd.ScreenFormat.CRCGCB, vzd.ScreenFormat.CBCGCR],
    [vzd.ScreenFormat.GRAY8, vzd.ScreenFormat.DOOM_256_COLORS8, vzd.ScreenFormat.RGB24, vzd.ScreenFormat.BGR24,
     vzd.ScreenFormat.RGBA32, vzd.ScreenFormat.ARGB32, vzd.ScreenFormat.BGRA32, vzd.ScreenFormat.ABGR32]
]

# A fixed set of colors for each potential label
# for rendering an image.
# 256 is not nearly enough for all IDs, but we limit
# ourselves here to avoid hogging too much memory.
LABEL_COLORS = np.random.default_rng(42).uniform(25, 256, size=(256, 3)).astype(np.uint8)

class DoomEnv(gym.Env):
    def __init__(self, config, frame_skip=1, frame_stack=1, down_sample=(None, None), to_torch=True, normalize=True,
                 multiple_buttons=False, add_depth=False, add_labels=False, add_automap=False, add_audio=False,
                 add_position_vars=False, add_health_vars=False, add_ammo_vars=False,
                 add_info_vars=[]):
        """
        :param config:               Specifies environment config
        :param frame_skip:           Skips frames, default: 1
        :param frame_stack:          Stacks frames, default: 1
        :param down_sample:          Down samples (BILINEAR) observation buffers, default: No down sampling
        :param to_torch:             Returns PyTorch Tensors over NumPy Arrays, default: True
        :param multiple_buttons:     Determines whether or not to one hot encode the action sent to the environment. If
                                     using delta buttons and/or you want to allow multiple buttons to be selected at
                                     once, set this to True and one hot encode the action yourself before calling step.
        :param add_depth:            Adds depth buffer to observation
        :param add_labels:           Adds labels buffer to observation
        :param add_automap:          Adds automap buffer to observation
        :param add_audio:            Adds audio buffer to observation
        :param add_position_vars:    Adds game variables specified by POSITIONS to observation
        :param add_health_vars:      Adds game variables specified by HEALTH to observation
        :param add_ammo_vars:        Adds game variables specified by AMMO to observation
        :param add_info_vars:        Adds game variables specified by the list to the info section
        """
        super().__init__()
        # save configuration
        config_path = path.join(BASE_PATH, config)
        self.config = config
        self.game = vzd.DoomGame()
        self.game.load_config(config_path)
        self.game.set_window_visible(False)
        self.window_surface = None
        self.isopen = True

        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        if down_sample[0] is None:
            down_sample = (self.game.get_screen_height(), down_sample[1])
        if down_sample[1] is None:
            down_sample = (down_sample[0], self.game.get_screen_width())
        self.down_sample = down_sample
        self.to_torch = to_torch
        self.normalize = normalize

        # Determine which buffers/variables to enable/already enabled via the config
        self.has_depth = add_depth or self.game.is_depth_buffer_enabled()
        self.has_labels = add_labels or self.game.is_labels_buffer_enabled()
        self.has_automap = add_automap or self.game.is_automap_buffer_enabled()
        self.has_audio = add_audio or self.game.is_audio_buffer_enabled()
        if add_position_vars:
            self.__add_game_variables(POSITIONS)
        if add_health_vars:
            self.__add_game_variables(HEALTH)
        if add_ammo_vars:
            self.__add_game_variables(AMMO)
        if len(add_info_vars) != 0:
            self.__add_game_variables(add_info_vars)
        # determine index of game variables
        num_position_vars, num_health_vars, num_ammo_vars, num_info_vars = self.__parse_game_variables(add_info_vars)
        self.has_position_vars = num_position_vars
        self.has_health_vars = num_health_vars
        self.has_ammo_vars = num_ammo_vars
        self.has_info_vars = num_info_vars
        self.info_vars = add_info_vars

        # set observation space
        self.screen_type = self.__get_screen_type()
        self.observation_space = self.__parse_observation_space()
        self.frames = deque([], maxlen=frame_stack)
        if isinstance(self.observation_space, spaces.Tuple):
            self.frames = [deque([], maxlen=frame_stack) for _ in range(len(self.observation_space))]
        self.lift_axis = 0
        if self.screen_type != 0:
            self.lift_axis = -1

        # set action space
        self.no_encode_action = self.__has_delta_button() or multiple_buttons
        self.action_space = self.__parse_game_actions()

        # init game
        try:
            self.game.init()
        except Exception as e:
            # Audio fix https://github.com/mwydmuch/ViZDoom/pull/486#issuecomment-895128340
            self.game.add_game_args("+snd_efx 0")
            self.game.init()

    def reset(self):
        self.game.new_episode()
        for i in range(self.frame_stack):
            self.__append_observation_buffers()
        observation = self.__get_observation_frames()
        return observation

    def step(self, action):
        info = self.__get_info()
        if not self.no_encode_action:
            action = self.__one_hot_encode_action(action)
        reward = self.game.make_action(action, self.frame_skip)
        done = int(self.game.is_episode_finished())
        self.__append_observation_buffers()
        observation = self.__get_observation_frames()
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
            return self.isopen

    def __build_human_render_image(self):
        """
        Stack all available buffers into one for human consumption
        """
        game_state = self.game.get_state()
        valid_buffers = game_state is not None

        if not valid_buffers:
            # Return a blank image
            num_enabled_buffers = 1 + self.has_depth + self.has_labels + self.has_automap
            img = np.zeros(
                (self.game.get_screen_height(), self.game.get_screen_width() * num_enabled_buffers, 3,), dtype=np.uint8
            )
            return img

        screen_buffer = game_state.screen_buffer
        screen_format = self.game.get_screen_format()
        if screen_format == vzd.ScreenFormat.GRAY8 or screen_format == vzd.ScreenFormat.DOOM_256_COLORS8:
            screen_buffer = np.repeat(np.expand_dims(screen_buffer, axis=self.lift_axis), repeats=3, axis=self.lift_axis)
        if screen_format == vzd.ScreenFormat.RGBA32 or screen_format == vzd.ScreenFormat.BGRA32:
            screen_buffer = screen_buffer[:, :, :3]
        if screen_format == vzd.ScreenFormat.ABGR32 or screen_format == vzd.ScreenFormat.ARGB32:
            screen_buffer = screen_buffer[:, :, 1:]
        image_list = [screen_buffer]

        if self.has_depth:
            image_list.append(np.repeat(np.expand_dims(game_state.depth_buffer, axis=self.lift_axis), repeats=3, axis=self.lift_axis))

        if self.has_labels:
            # Give each label a fixed color.
            # We need to connect each pixel in labels_buffer to the corresponding
            # id via `value``
            labels_rgb = np.zeros_like(screen_buffer)
            labels_buffer = game_state.labels_buffer
            for label in game_state.labels:
                color = LABEL_COLORS[label.object_id % 256]
                if self.screen_type == 0:
                    color = np.expand_dims(color, axis=-1)
                    labels_rgb[:, labels_buffer == label.value] = color
                else:
                    labels_rgb[labels_buffer == label.value] = color
            image_list.append(labels_rgb)

        if self.has_automap:
            automap = game_state.automap_buffer
            if screen_format == vzd.ScreenFormat.GRAY8 or screen_format == vzd.ScreenFormat.DOOM_256_COLORS8:
                automap = np.repeat(np.expand_dims(automap, axis=self.lift_axis), repeats=3, axis=self.lift_axis)
            if screen_format == vzd.ScreenFormat.RGBA32 or screen_format == vzd.ScreenFormat.BGRA32:
                automap = automap[:, :, :3]
            if screen_format == vzd.ScreenFormat.ABGR32 or screen_format == vzd.ScreenFormat.ARGB32:
                automap = automap[:, :, 1:]
            image_list.append(automap)

        if self.screen_type == 0:
            for i in range(len(image_list)):
                image_list[i] = image_list[i].transpose(1, 2, 0)  # CHW -> HWC
        return np.concatenate(image_list, axis=1)

    def close(self):
        if self.window_surface:
            pygame.quit()
            self.isopen = False

    def __exit__(self):
        self.close()

    def __append_observation_buffers(self):
        """
        Appends most recent frame to frame buffer/s
        Handles multiple buffers (i.e. observation space of Tuples)
        """
        if isinstance(self.observation_space, spaces.Tuple):
            observation = self.__get_single_observation()
            for i in range(len(self.observation_space)):
                self.frames[i].append(observation[i])
        else:
            self.frames.append(self.__get_single_observation())

    def __get_observation_frames(self):
        """
        Get stacked frames of observation
        Handles multiple buffers (i.e. observation space of Tuples) and in this case returns a tuple of stacked frames
        of observations
        """
        if isinstance(self.observation_space, spaces.Tuple):
            obs = []
            for i in range(len(self.observation_space)):
                obs.append(self.__to_torch(np.array(LazyFrames(list(self.frames[i])))))
            return tuple(obs)
        else:
            return self.__to_torch(np.array(LazyFrames(list(self.frames))))

    def __transform_single_observation(self, observation):
        """
        Down Samples if necessary and
        Lifts dimensions of observations with no channels to have a channel from HW->1HW//HW1 depending on what axis the
        channels are according to the screen buffer
        """
        if self.down_sample != (self.game.get_screen_height(), self.game.get_screen_width()):
            screen_format = self.game.get_screen_format()
            if len(observation.shape) > 2 and (screen_format == vzd.ScreenFormat.CRCGCB or screen_format == vzd.ScreenFormat.CBCGCR):
                observation = observation.transpose(1, 2, 0)  # CHW -> HWC
            observation = cv2.resize(observation, (self.down_sample[1], self.down_sample[0]), interpolation=cv2.INTER_LINEAR)
            if len(observation.shape) > 2 and (screen_format == vzd.ScreenFormat.CRCGCB or screen_format == vzd.ScreenFormat.CBCGCR):
                observation = observation.transpose(2, 0, 1)  # HWC -> CHW
        if len(observation.shape) == 2:
            observation = np.expand_dims(observation, axis=self.lift_axis)  # HW -> HW1//1HW
        return observation

    def __to_torch(self, observation):
        """
        Converts observation to PyTorch Tensor
        """
        if self.to_torch:
            observation = torch.from_numpy(observation)
        if self.normalize:
            observation = observation.float() / 255.
        return observation

    def __get_single_observation(self):
        """
        Gets the current observation (1 frame)
        """
        observation = []
        state = self.game.get_state()

        if state:
            screen_buffer = state.screen_buffer
            observation.append(self.__to_torch(self.__transform_single_observation(screen_buffer)))
            if self.has_depth:
                observation.append(self.__to_torch(self.__transform_single_observation(state.depth_buffer)))
            if self.has_labels:
                observation.append(self.__to_torch(self.__transform_single_observation(state.labels_buffer)))
            if self.has_automap:
                observation.append(self.__to_torch(self.__transform_single_observation(state.automap_buffer)))
            if self.has_audio:
                observation.append(self.__to_torch(state.audio_buffer))
            if self.has_position_vars:
                observation.append(self.__to_torch(np.array([state.game_variables[self.position_idxs]])))
            if self.has_health_vars:
                observation.append(self.__to_torch(np.array([state.game_variables[self.health_idxs]])))
            if self.has_health_vars:
                observation.append(self.__to_torch(np.array([state.game_variables[self.ammo_idxs]])))
        else:
            obs_space = self.observation_space
            # there is no state in the terminal step, so a "zero observation is returned instead"
            if isinstance(self.observation_space, gym.spaces.box.Box):
                obs_space = [obs_space]

            for space in obs_space:
                observation.append(self.__to_torch(np.zeros(space.shape[1:], dtype=space.dtype)))

        if len(observation) == 1:
            observation = observation[0]
        return observation

    def __get_info(self):
        """
        Get the info dictionary containing value of game variables given by self.info_vars
        """
        if self.has_info_vars:
            state = self.game.get_state()
            info = dict()
            for i in range(self.has_info_vars):
                info[str(self.info_vars[i])] = np.array([state.game_variables[self.info_idxs[i]]])
            return info
        else:
            return {}

    def __one_hot_encode_action(self, action):
        """
        :param action: index of which action to take, should be within the button range
        :return: List act, a one hot encoded list, where act[action] = 1
        """
        act = np.zeros(self.game.get_available_buttons_size(), dtype=np.uint8)
        act[action] = 1
        act = act.tolist()
        return act

    def __parse_observation_space(self):
        """
        :return: observation space of environment
        """
        # channels = C, shape: CHW//HWC
        observation_space = [self.__get_screen_box(self.game.get_screen_channels())]
        if self.has_depth:
            # channels = 1, shape: 1HW//HW1
            self.game.set_depth_buffer_enabled(True)
            observation_space.append(self.__get_screen_box(1))
        if self.has_labels:
            # channels = 1, shape: 1HW//HW1
            self.game.set_labels_buffer_enabled(True)
            observation_space.append(self.__get_screen_box(1))
        if self.has_automap:
            # channels = C, shape: CHW//HWC
            self.game.set_automap_buffer_enabled(True)
            self.game.set_automap_mode(vzd.AutomapMode.OBJECTS)
            self.game.set_automap_rotate(False)
            self.game.set_automap_render_textures(False)
            observation_space.append(self.__get_screen_box(self.game.get_screen_channels()))
        if self.has_audio:
            # C = 1260 audio buffer size, shape: C2
            self.game.set_audio_buffer_enabled(True)
            observation_space.append(spaces.Box(-1, 1, (self.frame_stack, 1260 * self.game.get_audio_buffer_size(), 2), dtype=np.uint16))

        if len(observation_space) == 1:
            return observation_space[0]
        else:
            return spaces.Tuple(observation_space)

    def __has_delta_button(self):
        """
        :return: True if game has delta button, False otherwise
        """
        buttons = self.game.get_available_buttons()
        for button in buttons:
            if vzd.is_delta_button(button):
                return True
        return False

    def __parse_game_actions(self):
        """
        :return: action space of environment, Box if delta button, Discrete if only binary buttons
        """
        if self.no_encode_action:
            return spaces.Box(-np.inf, np.inf, (self.game.get_available_buttons_size(),))
        else:
            return spaces.Discrete(self.game.get_available_buttons_size())

    def __get_box(self, height, width, channels=1, channels_first=True, dtype=np.uint8):
        """
        Helper function to return correct Box shape/type
        channels_first == True: NCHW
        channels_first == False: NHWC
        channels == 0: NHW
        """
        if channels != 0:
            if channels_first:
                return spaces.Box(0, 255, (self.frame_stack, channels, height, width), dtype=dtype)
            else:
                return spaces.Box(0, 255, (self.frame_stack, height, width, channels), dtype=dtype)
        else:
            return spaces.Box(0, 255, (self.frame_stack, height, width), dtype=dtype)

    def __get_screen_box(self, channels=0):
        """
        Helper function to return correct Box shape/type depending on screen type
        used to determine shape for screen, depth, labels, automap buffer
        """
        if self.screen_type == 0:
            return self.__get_box(self.down_sample[0], self.down_sample[1], channels)
        elif self.screen_type == 1:
            return self.__get_box(self.down_sample[0], self.down_sample[1], channels,
                           channels_first=False)
        else:
            return self.__get_box(self.down_sample[0], self.down_sample[1], channels,
                           channels_first=False)

    def __get_screen_type(self):
        """
        Determines which 'screen type' we have, 3 possible options indicated by SCREEN_FORMATS
        0 == CHW np.uint8
        1 == HWC np.uint8
        used to determine shape for screen, depth, labels, automap buffer
        """
        screen_format = self.game.get_screen_format()
        for i in range(len(SCREEN_FORMATS)):
            if screen_format in SCREEN_FORMATS[i]:
                return i
        return -1

    def __add_game_variables(self, variables):
        """
        adds variables to available game variables
        """
        for i in range(len(variables)):
            self.game.add_available_game_variable(variables[i])

    def __parse_game_variables(self, add_info_vars):
        """
        parses game variables to determine index of POSITIONS/HEALTH/AMMO and add_info_vars variables
        """
        game_variables = self.game.get_available_game_variables()

        self.position_idxs = np.array([False] * len(POSITIONS))
        self.health_idxs = np.array([False] * len(HEALTH))
        self.ammo_idxs = np.array([False] * len(AMMO))
        self.info_idxs = np.array([False] * len(add_info_vars))
        position_idx = 0
        health_idx = 0
        ammo_idx = 0
        info_idx = 0

        for i in range(len(game_variables)):
            game_var = game_variables[i]
            if game_var in POSITIONS:
                self.position_idxs[position_idx] = i
                position_idx += 1
            elif game_var in HEALTH:
                self.health_idxs[health_idx] = i
                health_idx += 1
            elif game_var in AMMO:
                self.ammo_idxs[ammo_idx] = i
                ammo_idx += 1
            # can be in both POSITIONS/HEALTH/AMMO and info section
            if game_var in add_info_vars:
                self.info_idxs[info_idx] = i
                info_idx += 1
        return position_idx, health_idx, ammo_idx, info_idx
