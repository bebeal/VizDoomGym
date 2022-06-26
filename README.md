# VizDoomGym
Highly Customizable Gym interface for ViZDoom.



https://user-images.githubusercontent.com/42706447/175821475-e6702247-0257-4d5c-aef7-351b7b5adfb5.mp4

<sub><sup>(Rendered Buffers: Screen, Depth, Label, Automap, Audio)</sub></sup>

## Install

```commandline
git clone https://github.com/bebeal/VizDoomGym
cd vizdoomgym
pip install .
```

## Features

* All ViZDoom Screen Formats (RGB24, GRAY8, CRCGCB, CBCGCR, RGBA32, ARGB32, BGRA32, ABGR32, DOOM_256_COLORS8)
* Multiple observation options and rendering
  * Screen Buffer
  * Depth Buffer
  * Labels Buffer
  * Automap Buffer
  * Audio Buffer
  * Any valid `GameVariable` ([List of GameVariables](https://github.com/mwydmuch/ViZDoom/blob/master/doc/Types.md#-gamevariable))
* Can mix and match multiple different buffers
* Can render any combination of buffers (even audio buffer)
* Frame Stacking
* Frame Skipping
* Down Sampling
* All possible buttons: Binary + Delta Buttons
* Can press/use any combination of buttons at once
* PyTorch Tensor Wrapper Option
* Image Buffer Normalization
* Probabilistically randomize scenario per call to `reset()` if given a list of scenarios
* Probabilistically randomize asset per call to `reset()` if given a list of assets
* Probabilistically randomize `doom_skill` i.e. difficulty per call to `reset()`


## Examples of Possible Action/Observation Spaces

* Example 0

```Python
env = DoomEnv("basic_g8_0000000_0_3.cfg", encode_action=True, no_single_channel=True, frame_stack=4)
```

Observation Space Shape: `(4, 240, 320)`

<img src="https://render.githubusercontent.com/render/math?math=\sim"/> Buffers: `(screen)`

Action Space: `Box([-inf -inf -inf], [inf inf inf], (3,), float32)`

Rendered View:

![0](https://user-images.githubusercontent.com/42706447/175820497-0bfe13b1-4d33-4916-b465-d92b35f4dc2c.png)

* Example 1

```Python
env = DoomEnv("basic_g8_0000000_4_4.cfg", encode_action=True, to_torch=True, add_labels_buffer=True, max_buttons_pressed=3, frame_stack=2)
```

Observation Space Shape: `(2, 1, 240, 320), (2, 1, 240, 320)`

<img src="https://render.githubusercontent.com/render/math?math=\sim"/> Buffers: `(screen, labels)`

Action Space: `Dict(binary:Discrete(15), continuous:Box([-inf -inf -inf -inf], [inf inf inf inf], (4,), float32))`

Rendered View:

![1](https://user-images.githubusercontent.com/42706447/175820499-411b9a30-2c01-4a33-b20b-96c6966a6478.png)


* Example 3

```Python
env = DoomEnv("basic_rgb_0000000_3_0.cfg", encode_action=True, to_torch=True, max_buttons_pressed=0, add_game_vars_buffer=[HEALTH, ARMOR], add_depth_buffer=True, add_audio_buffer=True)
```

Observation Space Shape: `(1, 3, 240, 320), (1, 1, 240, 320), (1, 3, 240, 320), (1, 5040, 2), (1, 2)`

<img src="https://render.githubusercontent.com/render/math?math=\sim"/> Buffers: `(screen, depth, automap, audio, game_vars(health, armor))`

Action Space: `MultiDiscrete([2 2 2])`

Rendered View:

![2](https://user-images.githubusercontent.com/42706447/175820511-773b6f45-dcb8-43e3-8462-b6a7f5c2e4fb.png)

## Parameter Documentation

```Python
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
:param image_size:          Resolution of image buffer, overrides value set in scenario file.
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
                                Action space can be a single one of binary/continuous action space, or a Tuple
                                containing both.
                                "binary"":
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
:param add_game_vars_buffer:Enables and adds these game_variables to the game state, passed back via an
                            observation buffer in `step()`. Default: ().
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
```

## TODO:
* Fix audio visualization
* Remove unnecessary storage of variables

## References
* [VizDoom](https://github.com/mwydmuch/ViZDoom)
* [Alternative VizDoomGym](https://github.com/shakenes/vizdoomgym)
