from pettingzoo.butterfly import pistonball_v4
import supersuit as ss
import os
from stable_baselines import PPO2
from array2gif import write_gif
import numpy as np
from pathlib import Path
import os
import sys

os.environ["SDL_VIDEODRIVER"] = "dummy"

path = sys.argv[1]

env = pistonball_v4.env()
env = ss.color_reduction_v0(env, mode='B')
env = ss.resize_v0(env, x_size=84, y_size=84)
env = ss.frame_stack_v1(env, 3)

reward_sum = 0
obs_list = []
i = 0
model = PPO2.load()
env.reset()

while True:
    for agent in env.agent_iter():
        observation, reward, done, info = env.last()
        reward_sum += reward
        action = model.predict(observation, deterministic=True)[0] if not done else None

        env.step(action)
        i += 1
        if i % (len(env.possible_agents)+1) == 0:
            obs_list.append(np.transpose(env.render(mode='rgb_array'), axes=(1, 0, 2)))
    env.close()
    break

name = os.path.basename(path).split()[0]

# print(reward)
print('writing gif')
write_gif(obs_list, str(Path.home())+'/gifs/'+name+'.gif', fps=15)
