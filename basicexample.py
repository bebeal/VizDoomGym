
import pygame
from vizdoomgym.vizdoomenv import DoomEnv
import gym

env = gym.make("VizdoomGymBasic-v0")
agent = lambda obs: env.action_space.sample()


obs = env.reset()  # (1, 3, 240, 320)
done = False
while not done:
    obs, reward, done, info = env.step(agent(obs))


env = DoomEnv("basic.cfg", image_size=(120, 160), to_torch=True, add_depth_buffer=True, add_labels_buffer=True, add_automap_buffer=True, frame_stack=4, frame_skip=4)
agent = lambda obs: env.action_space.sample()
obs = env.reset()  # ((4, 3, 120, 160), (4, 1, 120, 160), (4, 3, 120, 160)) ~ (screen, depth, labels, automap)
done = False
while not done:
    env.render()
    pygame.time.wait(10)
    obs, reward, done, info = env.step(agent(obs))
