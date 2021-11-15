
import random
from VizDoomEnv import DoomEnv

shoot = [1, 0, 0]
right = [0, 1, 0]
left = [0, 0, 1]
actions = [shoot, right, left]

env = DoomEnv(depth=True, labels=True, automap=True)
print(env.observation_space)
o = env.reset()

for t in range(1000):
    r, d, n_o, i = env.step(actions[random.randint(0, 2)])
    env.render()
    o = n_o
    if d:
        o = env.reset()

env = DoomEnv()
print(env.observation_space)
o = env.reset()

for t in range(1000):
    r, d, n_o, i = env.step(random.randint(0, 2), one_hot=True)
    env.render()
    o = n_o
    if d:
        o = env.reset()

