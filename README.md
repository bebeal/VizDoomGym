# VizDoomGym
Custom Gym Wrapper for ViZDoom

https://user-images.githubusercontent.com/42706447/172284590-5dc353e6-a9fd-468b-84db-0bb7cc8c1119.mp4

<sub><sup>(Buffers: Screen, Depth, Label, Automap)</sub></sup>

## Features

* All ViZDoom Screen Types
* Addition of multiple buffers
  * Depth Buffer
  * Labels Buffer
  * Automap Buffer
  * Audio Buffer
  * Custom Positions Buffer
    * `GameVariable.POSITION_X`
    * `GameVariable.POSITION_Y`
    * `GameVariable.POSITION_Z`
    * `GameVariable.ANGLE`
  * Custom Health Buffer
    * `GameVariable.HEALTH`
    * `GameVariable.ARMOR`
  * Custom Ammo Buffer
    * `GameVariable.AMMO0`
    * `GameVariable.AMMO1`
    * `GameVariable.AMMO2`
    * `GameVariable.AMMO3`
    * `GameVariable.AMMO4`
    * `GameVariable.AMMO5`
    * `GameVariable.AMMO6`
    * `GameVariable.AMMO7`
    * `GameVariable.AMMO8`
    * `GameVariable.AMMO9`
* Frame Skipping
* Frame Stacking
* Down Sampling
* ViZDoom Delta Buttons
* Multiple Buttons At Once
* PyTorch Tensor Wrapper

## Examples

### Basic Example

```Python
env = DoomEnv("basic.cfg")
```

Observation Space Shape: `(1, 3, 240, 320)`

Action Space: `Discrete(3)`, automatically one hot encodes it for you, just give index

### Delta Buttons Example

```Python
env = DoomEnv("basic_with_delta.cfg")  # Assumes environment has a delta button specified
```

Observation Space Shape: `(1, 3, 240, 320)`

Action Space: `Box(3)`, you must fully specify yourself (one hot encode it)

### Frame Stacking Example

```Python
env = DoomEnv("basic.cfg", frame_stack=4)
```

Observation Space Shape: `(4, 3, 240, 320)`

Action Space: `Discrete(3)`

### Multiple Buffers + Frame Stacking Example

```Python
env = DoomEnv("basic.cfg", add_depth=True, frame_stack=4)
```

Observation Space Shapes: `Tuple((4, 3, 240, 320), (4, 1, 240, 320))` <img src="https://render.githubusercontent.com/render/math?math=\sim"> Buffers: `Tuple(screen, depth)`

Action Space: `Discrete(3)`

### Down Sampling + Multiple Buffers + Frame Stacking

```Python
env = DoomEnv("basic.cfg", down_sample=(120, 160), add_depth=True, add_automap=True, add_audio=True, frame_stack=4) 
```

Observation Space Shapes: `Tuple((4, 3, 120, 160), (4, 1, 120, 160), (4, 3, 120, 160), (4, 5040, 2))` <img src="https://render.githubusercontent.com/render/math?math=\sim"> Buffers: `Tuple(screen, depth, automap, audio)`

Action Space: `Discrete(3)`


### Different ViZDoom Type (Gray Scaled) + Down Sampling + Multiple Buffers + Frame Stacking + Frame Skipping

```Python
env = DoomEnv("basic_GRAY8.cfg", down_sample=(120, 160), add_depth=True, add_automap=True, add_audio=True, frame_stack=4, frame_skip=4) 
```

Observation Space Shapes: `Tuple((4, 120, 160, 1), (4, 120, 160, 1), (4, 120, 160, 1), (4, 5040, 2))` <img src="https://render.githubusercontent.com/render/math?math=\sim"> Buffers: `Tuple(screen, depth, automap, audio)`

Action Space: `Discrete(3)`


### TODO:
* Better FrameStacking for multiple buffers, avoid loops
* Cleanup screen format stuff

### References
* [VizDoom](https://github.com/mwydmuch/ViZDoom)
* [Alternative VizDoomGym](https://github.com/shakenes/vizdoomgym)
