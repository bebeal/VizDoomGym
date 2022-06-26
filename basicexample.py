
import random
import numpy as np
import pygame
from vizdoomenv import DoomEnv

env = DoomEnv("basic.cfg", down_sample=(120, 160), to_torch=True, add_depth_buffer=True, add_labels_buffer=True, add_automap_buffer=True, frame_stack=4, frame_skip=4)
agent = lambda obs: env.action_space.sample()


obs = env.reset()  # ((4, 3, 120, 160), (4, 1, 120, 160), (4, 3, 120, 160)) ~ (screen, depth, labels, automap)
done = False
while not done:
    env.render()
    pygame.time.wait(50)
    obs, reward, done, info = env.step(agent(obs))
