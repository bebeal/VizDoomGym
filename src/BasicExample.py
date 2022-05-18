
import random
from VizDoomEnv import DoomEnv

env = DoomEnv("basic.cfg", down_sample=(120, 160), add_depth=True, add_automap=True, frame_stack=4)
observation = env.reset()  # ((4, 3, 120, 160), (4, 1, 120, 160), (4, 3, 120, 160)) ~ (screen, depth, automap)
for t in range(1000):
    reward, done, next_observation, info = env.step(random.randint(0, 2))
    env.render()
    observation = next_observation
    if done:
        observation = env.reset()
