
import random

import torch

from VizDoomEnv import DoomEnv
import vizdoom.vizdoom as vzd
import numpy as np
from gym import spaces


SCREEN_FORMATS = [
    [vzd.ScreenFormat.CRCGCB, vzd.ScreenFormat.CBCGCR],
    [vzd.ScreenFormat.GRAY8, vzd.ScreenFormat.DOOM_256_COLORS8], [vzd.ScreenFormat.RGB24, vzd.ScreenFormat.BGR24],
    [vzd.ScreenFormat.RGBA32, vzd.ScreenFormat.ARGB32, vzd.ScreenFormat.BGRA32, vzd.ScreenFormat.ABGR32]
]

expected_shapes = [
    [[3, 240, 320],
    [240, 320, 1], [240, 320, 3],
    [240, 320, 4]],
    [[3, 120, 160],
    [120, 160, 1], [120, 160, 3],
    [120, 160, 4]]
]

expected_types = [
    [np.uint8,
    np.uint8, np.uint8,
    np.uint8],
    [torch.uint8,
    torch.uint8, torch.uint8,
    torch.uint8]
]


def random_action(num_possible_actions):
    return random.randint(0, num_possible_actions)


def make_env(config, frame_stack=1, down_sample=(None, None), add_depth=False, add_labels=False, add_automap=False,
             add_audio=False, add_position_vars=False, add_health_vars=False, add_ammo_vars=False):
    return DoomEnv(config, frame_stack=frame_stack, down_sample=down_sample, add_depth=add_depth, add_labels=add_labels,
                   add_automap=add_automap, add_audio=add_audio, add_position_vars=add_position_vars,
                   add_health_vars=add_health_vars, add_ammo_vars=add_ammo_vars)


def test_observation_shape(env, expected_shape, expected_dtype):
    o = env.reset()
    assert o.shape == expected_shape
    assert o.dtype == expected_dtype


def test_observation_space_shape(env, expected_shape, expected_dtype):
    assert env.observation_space.shape == expected_shape
    assert env.observation_space.dtype == expected_dtype


def test_action_space(env, expected):
    assert env.action_space == expected


def observation_test(frame_stack, down_sample=(None, None), to_torch=True):
    n = 0
    for i in range(len(SCREEN_FORMATS)):
        for j in range(len(SCREEN_FORMATS[i])):
            env = make_env("basic" + str(n) + ".cfg", frame_stack=frame_stack, down_sample=down_sample)
            ds = down_sample != (None, None)
            expected_shape = np.zeros((frame_stack,) + tuple(expected_shapes[ds][i]))
            test_observation_space_shape(env, expected_shape=expected_shape.shape, expected_dtype=expected_types[0][i])
            if to_torch:
                expected_shape = torch.from_numpy(expected_shape)
            test_observation_shape(env, expected_shape=expected_shape.shape, expected_dtype=expected_types[to_torch][i])
            n += 1


def buffers_test(env, expected_shape, color, has_automap, has_depth, has_labels):
    assert env.observation_space[0].shape == expected_shape.shape  # NCHW//NHWC
    if has_automap:
        assert env.observation_space[has_depth + has_labels + has_automap].shape == expected_shape.shape  # NCHW//NHWC
    if color:
        expected_shape = np.zeros((expected_shape.shape[0], 1, expected_shape.shape[2], expected_shape.shape[3]))
    if has_depth:
        assert env.observation_space[has_depth].shape == expected_shape.shape  # N1HW//NHW1
    if has_labels:
        assert env.observation_space[has_depth + has_labels].shape == expected_shape.shape  # N1HW//NHW1

    o = env.reset()
    expected_shape = torch.from_numpy(expected_shape)
    if color:
        expected_shape = np.zeros((expected_shape.shape[0], 3, expected_shape.shape[2], expected_shape.shape[3]))
    assert o[0].shape == expected_shape.shape  # NCHW//NHWC
    if has_automap:
        assert o[has_depth + has_labels + has_automap].shape == expected_shape.shape  # NCHW//NHWC
    if color:
        expected_shape = np.zeros((expected_shape.shape[0], 1, expected_shape.shape[2], expected_shape.shape[3]))
    if has_depth:
        assert o[has_depth].shape == expected_shape.shape  # N1HW//NHW1
    if has_labels:
        assert o[has_depth + has_labels].shape == expected_shape.shape  # N1HW//NHW1


def screen_test():
    observation_test(1)
    print("Passed All Screen Type Tests")


def frame_stack_test():
    observation_test(4)
    print("Passed All Frame Stack Tests")


def down_sample_test():
    observation_test(1, down_sample=(120, 160))
    print("Passed All Down Sampled Tests")
    observation_test(4, down_sample=(120, 160))
    print("Passed All Down Sampled + Frame Stacked Tests")


def different_buffers_test():
    expected_shape = np.zeros((1,) + tuple(expected_shapes[0][0]))
    env = make_env(config="basic0.cfg", add_depth=True, add_labels=True, add_automap=True)
    buffers_test(env, expected_shape, color=True, has_automap=True, has_depth=True, has_labels=True)
    print("Passed Added Depth/Labels/Automap Buffer Tests")
    expected_shape = np.zeros((4,) + tuple(expected_shapes[1][1]))
    env = make_env(config="basic2.cfg", frame_stack=4, down_sample=(120, 160), add_depth=True, add_labels=True, add_automap=True)
    buffers_test(env, expected_shape, color=False, has_automap=True, has_depth=True, has_labels=True)
    print("Passed Stacked + Down Sampled + Gray Scaled + Added Depth/Labels/Automap Buffer Tests")


def action_space_test():
    env = make_env("basic0.cfg", )
    test_action_space(env, spaces.Discrete(3))
    env = make_env("basic1.cfg")
    test_action_space(env, spaces.Box(-np.inf, np.inf, (4,)))
    print("Passed All Action Space Tests")


def run_test():
    """
    Assumes `basic<#>.cfg` exists (in scenarios directory) from <#> range [0, 9] where each screen type corresponds to
    the ones in SCREEN_FORMATS and basic1.cfg has a delta action
    """
    screen_test()
    frame_stack_test()
    action_space_test()
    down_sample_test()
    different_buffers_test()


if __name__ == "__main__":
    run_test()
