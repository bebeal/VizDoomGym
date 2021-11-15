from os import path

import gym
from gym import spaces
from gym.spaces import Box, Discrete

import vizdoom.vizdoom as vzd

import numpy as np


turn_off_rendering = False
try:
    from gym.envs.classic_control import rendering
except Exception as e:
    print(e)
    turn_off_rendering = True

BASE_PATH = "scenarios"


class DoomEnv(gym.Env):
    def __init__(self, config="basic.cfg", **kwargs):
        super().__init__()
        config_path = path.join(BASE_PATH, config)
        self.config = config
        self.game = vzd.DoomGame()
        self.game.load_config(config_path)
        self.game.set_window_visible(False)

        self.position_indices = [0, 1, 2, 3]
        self.health_indices = [4, 5]
        self.ammo_indices = list(range(6, 15))
        position, health, ammo = self.__parse_game_variables()
        self.depth = kwargs.get("depth", False) or self.game.is_depth_buffer_enabled()
        self.labels = kwargs.get("labels", False) or self.game.is_labels_buffer_enabled()
        self.automap = kwargs.get("automap", False) or self.game.is_automap_buffer_enabled()
        self.audio = kwargs.get("audio", False) or self.game.is_audio_buffer_enabled()
        self.position = kwargs.get("position", False) or position
        self.health = kwargs.get("health", False) or health
        self.ammo = kwargs.get("ammo", False) or ammo

        self.action_space = Discrete(self.game.get_available_buttons_size())

        observation_space = [self.__screen_format_obs()]

        if self.depth:
            self.game.set_depth_buffer_enabled(True)
            observation_space.append(
                Box(0, 255, (
                    self.game.get_screen_height(),
                    self.game.get_screen_width()),
                    dtype=np.uint8)
            )

        if self.labels:
            self.game.set_labels_buffer_enabled(True)
            observation_space.append(
                Box(0, 255, (
                    self.game.get_screen_height(),
                    self.game.get_screen_width()),
                           dtype=np.uint8)
            )

        if self.automap:
            self.game.set_automap_buffer_enabled(True)
            observation_space.append(self.__screen_format_obs())

        if self.audio:
            self.game.set_audio_buffer_enabled(True)
            observation_space.append(Box(-1, 1, (5040, 2), dtype=np.uint16))

        if self.position:
            if not position:
                self.game.add_available_game_variable(vzd.GameVariable.POSITION_X)
                self.game.add_available_game_variable(vzd.GameVariable.POSITION_Y)
                self.game.add_available_game_variable(vzd.GameVariable.POSITION_Z)
                self.game.add_available_game_variable(vzd.GameVariable.ANGLE)
            observation_space.append(Box(-np.Inf, np.Inf, (4, 1)))

        if self.health:
            if not health:
                self.game.add_available_game_variable(vzd.GameVariable.HEALTH)
                self.game.add_available_game_variable(vzd.GameVariable.ARMOR)
            observation_space.append(Box(0, np.Inf, (2, 1)))

        if self.ammo:
            if not ammo:
                self.game.add_available_game_variable(vzd.GameVariable.AMMO0)
                self.game.add_available_game_variable(vzd.GameVariable.AMMO1)
                self.game.add_available_game_variable(vzd.GameVariable.AMMO2)
                self.game.add_available_game_variable(vzd.GameVariable.AMMO3)
                self.game.add_available_game_variable(vzd.GameVariable.AMMO4)
                self.game.add_available_game_variable(vzd.GameVariable.AMMO5)
                self.game.add_available_game_variable(vzd.GameVariable.AMMO6)
                self.game.add_available_game_variable(vzd.GameVariable.AMMO7)
                self.game.add_available_game_variable(vzd.GameVariable.AMMO8)
                self.game.add_available_game_variable(vzd.GameVariable.AMMO9)
            observation_space.append(Box(0, np.Inf, (10, 1)))

        if len(observation_space) == 1:
            self.observation_space = observation_space[0]
        else:
            self.observation_space = spaces.Tuple(observation_space)

        self.extra_info = kwargs.get("extra_info", False)
        if self.extra_info:
            self.game.add_available_game_variable(vzd.GameVariable.DEAD)
            self.game.add_available_game_variable(vzd.GameVariable.KILLCOUNT)
            self.game.add_available_game_variable(vzd.GameVariable.ITEMCOUNT)
            self.game.add_available_game_variable(vzd.GameVariable.HITCOUNT)
            self.game.add_available_game_variable(vzd.GameVariable.DAMAGECOUNT)
            self.game.add_available_game_variable(vzd.GameVariable.DAMAGE_TAKEN)
            self.game.add_available_game_variable(vzd.GameVariable.KILLCOUNT)

        self.viewer = None

        try:
            self.game.init()
        except e:
            # Audio fix https://github.com/mwydmuch/ViZDoom/pull/486#issuecomment-895128340
            self.game.add_game_args("+snd_efx 0")
            self.game.init()

    def reset(self):
        self.game.new_episode()
        return self.__get_observation()

    def step(self, action, one_hot=False):
        info = self.__get_info()
        if one_hot:
            action = self.__parse_action(action)
        reward = self.game.make_action(action)
        done = self.game.is_episode_finished()
        observation = self.__get_observation()
        return reward, done, observation, info

    def render(self, mode="human"):
        if turn_off_rendering:
            return
        try:
            img = self.game.get_state().screen_buffer
            img = np.transpose(img, [1, 2, 0])

            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
        except AttributeError:
            pass

    def close(self):
        if self.viewer:
            self.viewer.close()

    def __get_observation(self):
        observation = []
        state = self.game.get_state()

        if state is not None:
            observation.append(state.screen_buffer)
            if self.depth:
                observation.append(state.depth_buffer)
            if self.labels:
                observation.append(state.labels_buffer)
            if self.automap:
                observation.append(state.automap_buffer)
            if self.audio:
                observation.append(state.audio_buffer)
            if self.position:
                observation.append(np.array([state.game_variables[i] for i in self.position_indices]))
            if self.health:
                observation.append(np.array([state.game_variables[i] for i in self.health_indices]))
            if self.health:
                observation.append(np.array([state.game_variables[i] for i in self.ammo_indices]))
        else:
            # there is no state in the terminal step, so a "zero observation is returned instead"
            if isinstance(self.observation_space, gym.spaces.box.Box):
                # Box isn't iterable
                obs_space = [self.observation_space]
            else:
                obs_space = self.observation_space

            for space in obs_space:
                observation.append(np.zeros(space.shape, dtype=space.dtype))

        # if there is only one observation, return obs as array to sustain compatibility
        if len(observation) == 1:
            observation = observation[0]
        return observation

    def __parse_action(self, action):
        # one hot encode action
        act = np.zeros(self.action_space.n)
        act[action] = 1
        act = np.uint8(act)
        act = act.tolist()
        return act

    def __parse_game_variables(self):
        game_variables = self.game.get_available_game_variables()
        position = False
        health = False
        ammo = False

        for i in range(len(game_variables)):
            if game_variables[i] is vzd.GameVariable.POSITION_X:
                position = True
                self.position_indices[0] = i
            elif game_variables[i] is vzd.GameVariable.POSITION_Y:
                position = True
                self.position_indices[1] = i
            elif game_variables[i] is vzd.GameVariable.POSITION_Z:
                position = True
                self.position_indices[2] = i
            elif game_variables[i] is vzd.GameVariable.ANGLE:
                position = True
                self.position_indices[3] = i
            elif game_variables[i] is vzd.GameVariable.HEALTH:
                health = True
                self.health_indices[0] = i
            elif game_variables[i] is vzd.GameVariable.ARMOR:
                health = True
                self.health_indices[1] = i
            elif game_variables[i] is vzd.GameVariable.AMMO0:
                ammo = True
                self.ammo_indices[0] = i
            elif game_variables[i] is vzd.GameVariable.AMMO1:
                ammo = True
                self.ammo_indices[1] = i
            elif game_variables[i] is vzd.GameVariable.AMMO2:
                ammo = True
                self.ammo_indices[2] = i
            elif game_variables[i] is vzd.GameVariable.AMMO3:
                ammo = True
                self.ammo_indices[3] = i
            elif game_variables[i] is vzd.GameVariable.AMMO4:
                ammo = True
                self.ammo_indices[4] = i
            elif game_variables[i] is vzd.GameVariable.AMMO5:
                ammo = True
                self.ammo_indices[5] = i
            elif game_variables[i] is vzd.GameVariable.AMMO6:
                ammo = True
                self.ammo_indices[6] = i
            elif game_variables[i] is vzd.GameVariable.AMMO7:
                ammo = True
                self.ammo_indices[7] = i
            elif game_variables[i] is vzd.GameVariable.AMMO8:
                ammo = True
                self.ammo_indices[8] = i
            elif game_variables[i] is vzd.GameVariable.AMMO9:
                ammo = True
                self.ammo_indices[9] = i

        return position, health, ammo

    def __screen_format_obs(self):
        screen_format = self.game.get_screen_format()
        if  screen_format == vzd.ScreenFormat.CRCGCB or \
            screen_format == vzd.ScreenFormat.CBCGCR:
            return Box(0, 255, (
                    self.game.get_screen_channels(),
                    self.game.get_screen_height(),
                    self.game.get_screen_width()),
                    dtype=np.uint8)
        elif screen_format == vzd.ScreenFormat.RGB24 or \
             screen_format == vzd.ScreenFormat.BGR24:
            return Box(0, 255, (
                    self.game.get_screen_height(),
                    self.game.get_screen_width(),
                    self.game.get_screen_channels()),
                    dtype=np.uint8)
        elif screen_format == vzd.ScreenFormat.RGBA32 or \
             screen_format == vzd.ScreenFormat.ARGB32 or \
             screen_format == vzd.ScreenFormat.BGRA32 or \
             screen_format == vzd.ScreenFormat.ABGR32:
            return Box(0, 255, (
                    self.game.get_screen_height(),
                    self.game.get_screen_width(),
                    self.game.get_screen_channels()),
                    dtype=np.uint32)
        elif screen_format == vzd.ScreenFormat.GRAY8 or \
             screen_format == vzd.ScreenFormat.DOOM_256_COLORS8:
            return Box(0, 255, (
                    self.game.get_screen_height(),
                    self.game.get_screen_width()),
                    dtype=np.uint8)

    def __get_info(self):
        info = {
            "config": self.config,
            "screen format": self.game.get_screen_format(),
            "depth buffer enabled": self.depth,
            "labels buffer enabled": self.labels,
            "automap buffer enabled": self.automap,
            "audio buffer enabled": self.audio,
            "positions enabled": self.position,
            "health enabled": self.health,
            "observation space": self.observation_space,
            "action space": self.action_space,
        }
        game_variables = self.game.get_state().game_variables
        if game_variables is not None:
            info.update(dict(zip(game_variables, self.game.get_state().game_variables)))
        return info