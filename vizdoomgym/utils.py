# Taken from https://github.com/openai/gym/blob/master/gym/wrappers/frame_stack.py

from typing import Union
import numpy as np
import gym
import scipy.fftpack as fft
import matplotlib
matplotlib.rcParams.update({'figure.max_open_warning': 0})
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
from os import environ

from gym.error import DependencyNotInstalled

environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame



class LazyFrames:
    """Ensures common frames are only stored once to optimize memory use.
    To further reduce the memory use, it is optionally to turn on lz4 to compress the observations.
    Note:
        This object should only be converted to numpy array just before forward pass.
    """

    __slots__ = ("frame_shape", "dtype", "shape", "lz4_compress", "_frames")

    def __init__(self, frames: list, lz4_compress: bool = False):
        """Lazyframe for a set of frames and if to apply lz4.
        Args:
            frames (list): The frames to convert to lazy frames
            lz4_compress (bool): Use lz4 to compress the frames internally
        """
        self.frame_shape = tuple(frames[0].shape)
        self.shape = (len(frames),) + self.frame_shape
        self.dtype = frames[0].dtype
        if lz4_compress:
            try:
                from lz4.block import compress
            except ImportError:
                raise DependencyNotInstalled(
                    "lz4 is not installed, run `pip install gym[other]`"
                )

            frames = [compress(frame) for frame in frames]
        self._frames = frames
        self.lz4_compress = lz4_compress

    def __array__(self, dtype=None):
        """Gets a numpy array of stacked frames with specific dtype.
        Args:
            dtype: The dtype of the stacked frames
        Returns:
            The array of stacked frames with dtype
        """
        arr = self[:]
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    def __len__(self):
        """Returns the number of frame stacks.
        Returns:
            The number of frame stacks
        """
        return self.shape[0]

    def __getitem__(self, int_or_slice: Union[int, slice]):
        """Gets the stacked frames for a particular index or slice.
        Args:
            int_or_slice: Index or slice to get items for
        Returns:
            np.stacked frames for the int or slice
        """
        if isinstance(int_or_slice, int):
            return self._check_decompress(self._frames[int_or_slice])  # single frame
        return np.stack(
            [self._check_decompress(f) for f in self._frames[int_or_slice]], axis=0
        )

    def __eq__(self, other):
        """Checks that the current frames are equal to the other object."""
        return self.__array__() == other

    def _check_decompress(self, frame):
        if self.lz4_compress:
            from lz4.block import decompress

            return np.frombuffer(decompress(frame), dtype=self.dtype).reshape(
                self.frame_shape
            )
        return frame


class DirectionalAudioVisualizer:
    def __init__(self, sampling_rate, buffer_size, height, width, fill=False, both_channels=True):
        """
        reference: https://github.com/rctcwyvrn/py_spec/blob/master/py_spec.py
        """
        self.fill = fill
        self.both_channels = both_channels
        self.num_channels = 2
        self.sampling_rate = sampling_rate
        self.buffer_size = buffer_size
        self.t = 1/sampling_rate
        self.width = width
        self.height = height
        self.radius = 50
        self.max = 130
        self.min_scale = 0.10
        self.add = 5
        self.freq_range = 120
        if self.fill:
            self.freq_range = 50
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(subplot_kw=dict(polar=True), figsize=[width/100, height/100], dpi=100)
        self.x = np.array([2*np.pi/self.freq_range*i for i in range(self.freq_range + 1)])
        # plt.title("Audio Buffer Visualization")
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.axis("off")
        if not self.fill:
            plt.ylim(-1, self.max + self.radius + self.add)

    def get_val(self, audio_channel):
        res = fft.fft(audio_channel)
        res = abs(res)[:self.freq_range]
        res = np.append(res, [res[0]])
        # bound range
        res[res > self.max] = self.max
        # clip small values to 1
        relative_max = max(res)
        res[res < (relative_max * self.min_scale)] = 0
        res = res + self.radius
        return res

    def get_vals(self, audio_buffer):
        if len(audio_buffer.shape) >= 3:
            audio_buffer = np.mean(audio_buffer, axis=0)
        audio_l = audio_buffer[:, 1]
        res_l = self.get_val(audio_l)
        if self.both_channels:
            audio_r = audio_buffer[:, 0]
            res_r = self.get_val(audio_r)
            return res_l, res_r
        return res_l

    def step(self, audio_buffer):
        audio_buffer = np.array(audio_buffer).astype(np.float32) / float(2 ** 15)
        if self.both_channels:
            res_l, res_r = self.get_vals(audio_buffer)
        else:
            res_l = self.get_vals(audio_buffer)
        plt.style.use('dark_background')
        # plt.title("Audio Buffer Visualization")
        plt.grid(False)
        plt.axis("off")
        if self.fill:
            if self.both_channels:
                self.ax.bar(x=self.x, height=res_r, width=0.08, bottom=0, color="blue", alpha=0.5, label="right channel")
            self.ax.bar(x=self.x, height=res_l, width=0.08, bottom=0, color="red", alpha=0.5, label="left channel")
            plt.ylim(-1, self.max + self.radius + self.add)
        else:
            if self.both_channels:
                self.ax.plot(self.x, res_r, color="blue", alpha=0.45, label="right channel")
                self.ax.plot(self.x, res_r + self.add, color="darkblue", alpha=0.65)
            self.ax.plot(self.x, res_l, color="red", alpha=0.45, label="left channel")
            self.ax.plot(self.x, res_l + self.add, color="darkred", alpha=0.65)
        # self.ax.set_theta_offset(np.deg2rad(90))
        if self.both_channels:
            plt.legend(loc="upper right", fontsize=7, bbox_to_anchor=(1.35, 1.05),)
        canvas = agg.FigureCanvasAgg(self.fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        surf = pygame.image.fromstring(raw_data, canvas.get_width_height(), "RGB")
        plt.cla()
        return pygame.surfarray.array3d(surf).transpose(1, 0, 2)

    def __exit__(self):
        del self.fig
        del self.ax
        plt.close("all")


class FrequencyAudioVisualizer:
    def __init__(self, sampling_rate, buffer_size, height, width, both_channels=False):
        self.num_channels = 2
        self.both_channels = both_channels
        self.sampling_rate = sampling_rate
        self.buffer_size = buffer_size
        self.width = width
        self.height = height
        plt.style.use("dark_background")
        self.fig, self.ax = plt.subplots(figsize=[width/100, height/100], dpi=100)
        self.n_samples = 1260 * self.buffer_size
        self.t_audio = self.n_samples / self.sampling_rate
        self.times = np.linspace(0, self.n_samples / self.sampling_rate, num=self.n_samples)
        plt.xlim(0, self.t_audio)
        plt.title("Audio Buffer Visualization")
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.axis("off")
        plt.ylim(-2, 2)

    def get_vals(self, audio_buffer):
        if len(audio_buffer.shape) >= 3:
            audio_buffer = np.mean(audio_buffer, axis=0)
        audio_l = audio_buffer[:, 0]
        if self.both_channels:
            audio_r = audio_buffer[:, 1]
            return audio_l, audio_r
        return audio_l

    def step(self, audio_buffer):
        audio_buffer = np.array(audio_buffer).astype(np.float32) / float(2 ** 15)
        plt.style.use("dark_background")
        plt.xlim(0, self.t_audio)
        plt.title("Audio Buffer Visualization")
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.axis("off")
        plt.ylim(-2, 2)
        if self.both_channels:
            res_l, res_r = self.get_vals(audio_buffer)
            plt.plot(self.times, res_r.reshape(-1), color="blue", alpha=0.5, label="right channel")
        else:
            res_l = self.get_vals(audio_buffer)
        plt.plot(self.times, res_l.reshape(-1), color="red", alpha=0.5, label="left channel")
        if self.both_channels:
            plt.legend(loc="upper right", fontsize=7)
        canvas = agg.FigureCanvasAgg(self.fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        surf = pygame.image.fromstring(raw_data, canvas.get_width_height(), "RGB")
        plt.cla()
        return pygame.surfarray.array3d(surf).transpose(1, 0, 2)

    def __exit__(self):
        del self.fig
        del self.ax
        plt.close("all")


def get_box(frame_stack, height, width, channels=1, channels_first=True, dtype=np.uint8):
    """
    Helper function to return correct Box shape/type
    possible shapes:
    HW
    CHW
    HWC
    NCHW
    NHWC
    NHW
    """
    shape = np.zeros((height, width))
    if channels:
        lift_axis = -1
        if channels_first:
            lift_axis = 0
        shape = np.repeat(np.expand_dims(shape, axis=lift_axis), repeats=channels, axis=lift_axis)

    if frame_stack:
        shape = np.repeat(np.expand_dims(shape, axis=0), repeats=frame_stack, axis=0)
    return gym.spaces.Box(0, 255, shape.shape, dtype=dtype)

