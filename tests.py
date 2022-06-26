import os
import random
import torch
import time
from vizdoomenv import DoomEnv
import os
import numpy as np
from gym.spaces import Box, MultiDiscrete, Discrete, Tuple, Dict

test_env_configs = os.path.abspath("./test_scenarios")


def get_scenario_path(scenario_name):
    return os.path.join(test_env_configs, scenario_name + ".cfg")


def test_observation_space(env, expected_shape, expected_types=np.uint8, to_torch=False):
    if isinstance(expected_shape[0], int):
        expected_observation_space = Box(0, 255, shape=expected_shape, dtype=expected_types)
    else:
        if not isinstance(expected_types, list):
            expected_types = [expected_types for i in range(len(expected_shape))]
        min_value = 0
        max_value = 255
        sound_min_value = -1
        sound_max_value = 1
        pha_buffers_min_value = -np.inf
        pha_buffers_max_value = np.inf
        expected_observation_space = Tuple([Box(sound_min_value if expected_shape[i][1:]==(1260*4, 2) else min_value if len(expected_shape[i]) > 2 else pha_buffers_min_value, sound_max_value if expected_shape[i][1:]==(1260*4, 2) else max_value if len(expected_shape[i]) > 2 else pha_buffers_max_value, shape=expected_shape[i], dtype=expected_types[i]) for i in range(len(expected_shape))])

    obs = env.reset()
    if isinstance(expected_observation_space, Tuple):
        for i in range(len(obs)):
            assert obs[i].shape == expected_shape[i], f"Incorrect observation shape: {obs[i].shape!r}" \
                                                      f"\nShould be: {expected_shape[i]}"
    else:
        assert obs.shape == expected_shape, f"Incorrect observation shape: {obs.shape!r}" \
                                            f"\nShould be: {expected_shape}"
    assert env.observation_space == expected_observation_space, f"Incorrect observation space:\n{env.observation_space!r}" \
                                                                f"\nShould be:\n{expected_observation_space!r}"
    # if test_step:
    if to_torch:
        if isinstance(expected_observation_space, Tuple):
            new_obs = []
            for i in range(len(obs)):
                new_obs.append(np.array(obs[i]))
            obs = tuple(new_obs)
        else:
            obs = np.array(obs)
        assert env.observation_space.contains(obs), f"Step observation:\n{obs!r}\nNot in space:\n{env.observation_space}"


def test_action_space(env, expected_action_space):
    assert env.action_space == expected_action_space, f"Incorrect action space: {env.action_space!r}, " \
                                                      f"should be: {expected_action_space!r}"
    obs = env.reset()
    # check successful call to step using action_space.sample()
    action = env.action_space.sample()
    env.step(action)


def test_passed():
    print("==== Test Passed ====\n")


def get_env(screen_type, buffers, num_binary, num_delta, to_torch=False, no_single_channel=False, frame_stack=1, down_sample=(None, None), max_buttons_pressed=0, encode_action=False):
    add_depth_buffer = False
    if int(buffers[0]):
        add_depth_buffer = True
    add_labels_buffer = False
    if int(buffers[1]):
        add_labels_buffer = True
    add_automap_buffer = False
    if int(buffers[2]):
        add_automap_buffer = True
    add_audio_buffer = False
    if int(buffers[3]):
        add_audio_buffer = True
    add_position_buffer = False
    if int(buffers[4]):
        add_position_buffer = True
    add_health_buffer = False
    if int(buffers[5]):
        add_health_buffer = True
    add_ammo_buffer = False
    if int(buffers[6]):
        add_ammo_buffer = True
    env_name = "basic_" + screen_type + "_0000000" + "_" + str(num_binary) + "_" + str(num_delta)
    return DoomEnv(scenarios=get_scenario_path(env_name), to_torch=to_torch, no_single_channel=no_single_channel,
                   frame_stack=frame_stack, down_sample=down_sample, add_depth_buffer=add_depth_buffer,
                   add_labels_buffer=add_labels_buffer, add_automap_buffer=add_automap_buffer,
                   add_audio_buffer=add_audio_buffer, add_position_buffer=add_position_buffer,
                   add_health_buffer=add_health_buffer, add_ammo_buffer=add_ammo_buffer,
                   encode_action=encode_action, max_buttons_pressed=max_buttons_pressed)


def test_multiple_buffer_observation_space():
    """
    Testing various observation spaces with multiple buffers activated
    Testing multiple combinations of the following options:
    All possible screen types `(GRAY8|RGB24)`
    no_single_channel = `(False, True)`
    frame_stack = `(1, 4)`
    buffers = `(depth, depth+labels, depth+labels+automap, depth+labels+automap+audio,
                depth+labels+automap+audio+position, depth+labels+automap+audio+position+health,
                depth+labels+automap+audio+position+health+ammo, depth+automap+position+ammo,
                labels+audio+health)`
    """
    print("==== Testing possible observation space shapes for multiple buffers ====")
    buffer_shapes = [
        # shapes when to_torch=False
        [
            # shapes when rgb
            [
                # shapes when no_single_channel=False
                [
                    # shapes when frame_stack == 1
                    [
                        (1, 240, 320, 3),
                        (1, 240, 320, 1),
                        (1, 240, 320, 1),
                        (1, 240, 320, 3),
                        (1, 1260*4, 2),
                        (1, 4),
                        (1, 2),
                        (1, 10)
                    ],
                    # shapes when frame_stack == 4
                    [
                        (4, 240, 320, 3),
                        (4, 240, 320, 1),
                        (4, 240, 320, 1),
                        (4, 240, 320, 3),
                        (4, 1260*4, 2),
                        (4, 4),
                        (4, 2),
                        (4, 10)
                    ]
                ],
                # shapes when no_single_channel=True (should enforce (NCHW|NHWC) -> NHW if C==1)
                [
                    # shapes when frame_stack == 1
                    [
                        (1, 240, 320, 3),
                        (1, 240, 320),
                        (1, 240, 320),
                        (1, 240, 320, 3),
                        (1, 1260*4, 2),
                        (1, 4),
                        (1, 2),
                        (1, 10)
                    ],
                    # shapes when frame_stack == 4
                    [
                        (4, 240, 320, 3),
                        (4, 240, 320),
                        (4, 240, 320),
                        (4, 240, 320, 3),
                        (4, 1260*4, 2),
                        (4, 4),
                        (4, 2),
                        (4, 10)
                    ]
                ]
            ],
            # shapes when g8
            [
                # shapes when no_single_channel=False
                [
                    # shapes when frame_stack == 1
                    [
                        (1, 240, 320, 1),
                        (1, 240, 320, 1),
                        (1, 240, 320, 1),
                        (1, 240, 320, 1),
                        (1, 1260*4, 2),
                        (1, 4),
                        (1, 2),
                        (1, 10)
                    ],
                    # shapes when frame_stack == 4
                    [
                        (4, 240, 320, 1),
                        (4, 240, 320, 1),
                        (4, 240, 320, 1),
                        (4, 240, 320, 1),
                        (4, 1260*4, 2),
                        (4, 4),
                        (4, 2),
                        (4, 10)
                    ]
                ],
                # shapes when no_single_channel=True (should enforce (NCHW|NHWC) -> NHW if C==1)
                [
                    # shapes when frame_stack == 1
                    [
                        (1, 240, 320),
                        (1, 240, 320),
                        (1, 240, 320),
                        (1, 240, 320),
                        (1, 1260*4, 2),
                        (1, 4),
                        (1, 2),
                        (1, 10)
                    ],
                    # shapes when frame_stack == 4
                    [
                        (4, 240, 320),
                        (4, 240, 320),
                        (4, 240, 320),
                        (4, 240, 320),
                        (4, 1260*4, 2),
                        (4, 4),
                        (4, 2),
                        (4, 10)
                    ]
                ]
            ]
        ],
        # shapes when to_torch=True
        [
            # shapes when rgb
            [
                # shapes when no_single_channel=False
                [
                    # shapes when frame_stack == 1
                    [
                        (1, 3, 240, 320),
                        (1, 1, 240, 320),
                        (1, 1, 240, 320),
                        (1, 3, 240, 320),
                        (1, 1260*4, 2),
                        (1, 4),
                        (1, 2),
                        (1, 10)
                    ],
                    # shapes when frame_stack == 4
                    [
                        (4, 3, 240, 320),
                        (4, 1, 240, 320),
                        (4, 1, 240, 320),
                        (4, 3, 240, 320),
                        (4, 1260*4, 2),
                        (4, 4),
                        (4, 2),
                        (4, 10)
                    ]
                ],
                # shapes when no_single_channel=True (should enforce (NCHW|NHWC) -> NHW if C==1)
                [
                    # shapes when frame_stack == 1
                    [
                        (1, 3, 240, 320),
                        (1, 240, 320),
                        (1, 240, 320),
                        (1, 3, 240, 320),
                        (1, 1260*4, 2),
                        (1, 4),
                        (1, 2),
                        (1, 10)
                    ],
                    # shapes when frame_stack == 4
                    [
                        (4, 3, 240, 320),
                        (4, 240, 320),
                        (4, 240, 320),
                        (4, 3, 240, 320),
                        (4, 1260*4, 2),
                        (4, 4),
                        (4, 2),
                        (4, 10)
                    ]
                ]
            ],
            # shapes when g8
            [
                # shapes when no_single_channel=False
                [
                    # shapes when frame_stack == 1
                    [
                        (1, 1, 240, 320),
                        (1, 1, 240, 320),
                        (1, 1, 240, 320),
                        (1, 1, 240, 320),
                        (1, 1260*4, 2),
                        (1, 4),
                        (1, 2),
                        (1, 10)
                    ],
                    # shapes when frame_stack == 4
                    [
                        (4, 1, 240, 320),
                        (4, 1, 240, 320),
                        (4, 1, 240, 320),
                        (4, 1, 240, 320),
                        (4, 1260*4, 2),
                        (4, 4),
                        (4, 2),
                        (4, 10)
                    ]
                ],
                # shapes when no_single_channel=True (should enforce (NCHW|NHWC) -> NHW if C==1)
                [
                    # shapes when frame_stack == 1
                    [
                        (1, 240, 320),
                        (1, 240, 320),
                        (1, 240, 320),
                        (1, 240, 320),
                        (1, 1260*4, 2),
                        (1, 4),
                        (1, 2),
                        (1, 10)
                    ],
                    # shapes when frame_stack == 4
                    [
                        (4, 240, 320),
                        (4, 240, 320),
                        (4, 240, 320),
                        (4, 240, 320),
                        (4, 1260*4, 2),
                        (4, 4),
                        (4, 2),
                        (4, 10)
                    ]
                ]
            ]
        ]
    ]
    for to_torch in [False, True]:
        for screen_type in ["rgb", "g8"]:
            for no_single_channel in [False, True]:
                for frame_stack in [1, 4]:
                    for buffers in ["1000000", "1100000", "1110000", "1111000", "1111100", "1111110", "1111111", "1010101",
                                    "0101010"]:
                        env = get_env(screen_type, buffers, 3, 0, to_torch=to_torch,
                                      no_single_channel=no_single_channel, frame_stack=frame_stack)
                        buf_shapes = buffer_shapes[to_torch][screen_type == "g8"][no_single_channel][frame_stack == 4]
                        expected_shape = [buf_shapes[0]]
                        expected_shape += [buf_shapes[i+1] for i in range(len(buffers)) if int(buffers[i])]
                        if to_torch:
                            types = [np.float32]
                            types += [np.float32 for i in range(len(buffers)) if int(buffers[i])]
                        else:
                            types = [np.uint8]
                            types += [np.uint8 if (i <= 2) else np.float32 if (i >= 4) else np.int16 for i in range(len(buffers)) if int(buffers[i])]
                        test_observation_space(env, tuple(expected_shape), expected_types=types, to_torch=to_torch)
    test_passed()


def test_screen_types_observation_space():
    """
    Testing various observation spaces
    All unique combinations of the following options:
    All possible screen types `(GRAY8|DOOM_256_COLORS8|RGB24|BGR24|CRCGCB|CBCGCR|RGBA32|ARGB32|BGRA32|ABGR32)`
    height,width = `((None, None), (200, 300))`
    to_torch = `(False, True)`
    no_single_channel = `(False, True)`
    frame_stack = `(1, 4)`
    """
    print("==== Testing all possible observation space shapes ====")
    SCREEN_TYPES = [
        # 1 channel HW1 by default
        ["g8", "256"],
        # 3 channel HW3 by default
        ["rgb", "bgr"],
        # 3 channel 3HW by default
        ["cr", "cb"],
        # 4 channel HW4
        ["rgba", "argb", "bgra", "abgr"]
    ]
    # combine all possible options for to_torch=(False, True), no_single_channel=(False, True),
    # and two different number of frame_stack=(1, 4), for all SCREEN_TYPES
    OBSERVATION_SHAPES = [
        # shapes for height=240, width=320
        [
            # shapes when to_torch=False
            [
                # shapes when no_single_channel=False
                [
                    # shapes when frame_stack == 1
                    [
                        (1, 240, 320, 1),
                        (1, 240, 320, 3),
                        (1, 3, 240, 320),
                        (1, 240, 320, 4)
                    ],
                    # shapes when frame_stack == 4
                    [
                        (4, 240, 320, 1),
                        (4, 240, 320, 3),
                        (4, 3, 240, 320),
                        (4, 240, 320, 4)
                    ]
                ],
                # shapes when no_single_channel=True (should encforce (NCHW|NHWC) -> NHW if C==1)
                [
                    # shapes when frame_stack == 1
                    [
                        (1, 240, 320),
                        (1, 240, 320, 3),
                        (1, 3, 240, 320),
                        (1, 240, 320, 4)
                    ],
                    # shapes when frame_stack == 4
                    [
                        (4, 240, 320),
                        (4, 240, 320, 3),
                        (4, 3, 240, 320),
                        (4, 240, 320, 4)
                    ]
                ]
            ],
            # shapes when to_torch=True (should always enforce (NCHW|NHW) regardless of screen type defined)
            [
                # shapes when no_single_channel=False
                [
                    # shapes when frame_stack == 1
                    [
                        (1, 1, 240, 320),
                        (1, 3, 240, 320),
                        (1, 3, 240, 320),
                        (1, 4, 240, 320)
                    ],
                    # shapes when frame_stack == 4
                    [
                        (4, 1, 240, 320),
                        (4, 3, 240, 320),
                        (4, 3, 240, 320),
                        (4, 4, 240, 320)
                    ]
                ],
                # shapes when no_single_channel=True (should encforce (NCHW|NHWC) -> NHW if C==1)
                [
                    # shapes when frame_stack == 1
                    [
                        (1, 240, 320),
                        (1, 3, 240, 320),
                        (1, 3, 240, 320),
                        (1, 4, 240, 320)
                    ],
                    # shapes when frame_stack == 4
                    [
                        (4, 240, 320),
                        (4, 3, 240, 320),
                        (4, 3, 240, 320),
                        (4, 4, 240, 320)
                    ]
                ]
            ]
        ],
        # shapes for height=200, width=300
        [
            # shapes when to_torch=False
            [
                # shapes when no_single_channel=False
                [
                    # shapes when frame_stack == 1
                    [
                        (1, 200, 300, 1),
                        (1, 200, 300, 3),
                        (1, 3, 200, 300),
                        (1, 200, 300, 4)
                    ],
                    # shapes when frame_stack == 4
                    [
                        (4, 200, 300, 1),
                        (4, 200, 300, 3),
                        (4, 3, 200, 300),
                        (4, 200, 300, 4)
                    ]
                ],
                # shapes when no_single_channel=True (should encforce (NCHW|NHWC) -> NHW if C==1)
                [
                    # shapes when frame_stack == 1
                    [
                        (1, 200, 300),
                        (1, 200, 300, 3),
                        (1, 3, 200, 300),
                        (1, 200, 300, 4)
                    ],
                    # shapes when frame_stack == 4
                    [
                        (4, 200, 300),
                        (4, 200, 300, 3),
                        (4, 3, 200, 300),
                        (4, 200, 300, 4)
                    ]
                ]
            ],
            # shapes when to_torch=True (should always enforce (NCHW|NHW) regardless of screen type defined)
            [
                # shapes when no_single_channel=False
                [
                    # shapes when frame_stack == 1
                    [
                        (1, 1, 200, 300),
                        (1, 3, 200, 300),
                        (1, 3, 200, 300),
                        (1, 4, 200, 300)
                    ],
                    # shapes when frame_stack == 4
                    [
                        (4, 1, 200, 300),
                        (4, 3, 200, 300),
                        (4, 3, 200, 300),
                        (4, 4, 200, 300)
                    ]
                ],
                # shapes when no_single_channel=True (should encforce (NCHW|NHWC) -> NHW if C==1)
                [
                    # shapes when frame_stack == 1
                    [
                        (1, 200, 300),
                        (1, 3, 200, 300),
                        (1, 3, 200, 300),
                        (1, 4, 200, 300)
                    ],
                    # shapes when frame_stack == 4
                    [
                        (4, 200, 300),
                        (4, 3, 200, 300),
                        (4, 3, 200, 300),
                        (4, 4, 200, 300)
                    ]
                ]
            ]
        ]
    ]
    for down_sample in [(None, None), (200, 300)]:
        for to_torch in [False, True]:
            for no_single_channel in [False, True]:
                for frame_stack in [1, 4]:
                    for screen_type_index in range(len(SCREEN_TYPES)):
                        for screen_type in SCREEN_TYPES[screen_type_index]:
                            env = get_env(screen_type, "0000000", 3, 0, to_torch, no_single_channel, frame_stack, down_sample=down_sample)
                            down_sample_index = down_sample == (200, 300)
                            frame_stack_index = frame_stack == 4
                            test_observation_space(env, OBSERVATION_SHAPES[down_sample_index][to_torch][no_single_channel][frame_stack_index][screen_type_index], np.float32 if to_torch else np.uint8, to_torch)

    test_passed()


def test_terminal_state():
    """
    Testing observation on terminal state (terminal state is handled differently)
    """
    print("==== Testing terminal state ====")
    screen_types = ["rgb", "g8"]
    for screen_type in screen_types:
        env = get_env(screen_type, "0000000", 3, 0)

        agent = lambda obs: env.action_space.sample()
        obs = env.reset()
        done = False
        while not done:
            a = agent(obs)
            (obs, _reward, done, _info) = env.step(a)
            if done:
                break
            env.close()
    test_passed()


def test_all_action_spaces():
    """
    Testing various action spaces:
    for max_buttons_pressed: 0
        num_binary_buttons: 0, num_delta_buttons: 3
        num_binary_buttons: 3, num_delta_buttons: 0
        num_binary_buttons: 4, num_delta_buttons: 4
    for max_buttons_pressed: 2
        num_binary_buttons: 0, num_delta_buttons: 3
        num_binary_buttons: 3, num_delta_buttons: 0
        num_binary_buttons: 4, num_delta_buttons: 4
    """
    print("==== Testing all action spaces ====")
    num_binary_buttons = [0, 3, 4]
    num_delta_buttons = [3, 0, 4]
    EXPECTED_ACTION_SPACES = [
        # max_buttons_pressed=0
        [
            Box(-np.inf, np.inf, (3,), np.float32),
            MultiDiscrete([2] * 3),
            Dict({"binary": MultiDiscrete([2] * 4), "continuous": Box(-np.inf, np.inf, (4,), np.float32)})
        ],
        # max_buttons_pressed=2
        [
            Box(-np.inf, np.inf, (3,), np.float32),
            Discrete(7),
            Dict({"binary": Discrete(11), "continuous": Box(-np.inf, np.inf, (4,), np.float32)})
        ]
    ]
    for max_buttons_pressed in [0, 2]:
        for button_idx in range(len(num_binary_buttons)):
            env = get_env("g8", "0000000", num_binary_buttons[button_idx], num_delta_buttons[button_idx], max_buttons_pressed=max_buttons_pressed, encode_action=True)
            test_action_space(env, EXPECTED_ACTION_SPACES[max_buttons_pressed==2][button_idx])

    test_passed()


def render_test():
    """
    Visual test for various rendering options
    """
    SCREEN_TYPES = [
        # 1 channel HW1 by default
        "g8", "256",
        # 3 channel HW3 by default
        "rgb", "bgr",
        # 3 channel 3HW by default
        "cr", "cb",
        # 4 channel HW4
        "rgba", "argb", "bgra", "abgr"
    ]
    # various screen types
    for screen_type in SCREEN_TYPES:
        print(screen_type)
        env = get_env(screen_type, "0000000", 3, 0, max_buttons_pressed=0, encode_action=True)

        agent = lambda obs: env.action_space.sample()
        done = False
        obs = env.reset()
        while not done:
            env.render()
            time.sleep(0.01)
            a = agent(obs)
            (obs, _reward, done, _info) = env.step(a)
        env.close()

    # multiple buffers
    for screen_type in ["g8", "rgb"]:
        for buffers in ["1000000", "1100000", "1110000", "1111000"]:
            print(screen_type, buffers)
            env = get_env(screen_type, buffers, 3, 0, max_buttons_pressed=0, encode_action=True)
            agent = lambda obs: env.action_space.sample()
            done = False
            obs = env.reset()
            while not done:
                env.render()
                time.sleep(0.01)
                a = agent(obs)
                (obs, _reward, done, _info) = env.step(a)
            env.close()


if __name__ == "__main__":
    test_screen_types_observation_space()
    test_multiple_buffer_observation_space()
    test_terminal_state()
    test_all_action_spaces()
    # render_test()
