
import random

import numpy as np

from VizDoomEnv import DoomEnv
import matplotlib.pyplot as plt

env = DoomEnv("basic.cfg", down_sample=(120, 160), add_depth=True, add_automap=True, add_labels=True, frame_stack=4, frame_skip=4)

observation = env.reset()  # ((4, 3, 120, 160), (4, 1, 120, 160), (4, 3, 120, 160)) ~ (screen, depth, automap)
for t in range(1000):
    next_observation, reward, done, info = env.step(random.randint(0, 2))
    env.render()
    plt.pause(0.1)
    observation = next_observation
    if done:
        observation = env.reset()
