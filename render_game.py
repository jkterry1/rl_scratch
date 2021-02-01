import ray
import pickle5 as pickle
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer
from pettingzoo.butterfly import pistonball_v3
import supersuit as ss
from ray.rllib.env import PettingZooEnv
from array2gif import write_gif
from ray.rllib.models import ModelCatalog
from run_rllib import MLPModelV2
import numpy as np
import os

os.environ["SDL_VIDEODRIVER"] = "dummy"

checkpoint_path = "/home/justinkterry/ray_results/pistonball_v3/PPO/PPO_pistonball_v3_53bee_00000_0_2021-02-01_04-24-08/checkpoint_160/checkpoint-160"

ModelCatalog.register_custom_model("MLPModelV2", MLPModelV2)


def env_creator(args):
    env = pistonball_v3.env(n_pistons=20, local_ratio=0, time_penalty=-0.1, continuous=True, random_drop=True, random_rotate=True, ball_mass=0.75, ball_friction=0.3, ball_elasticity=1.5, max_cycles=125)
    env = ss.color_reduction_v0(env, mode='B')
    env = ss.dtype_v0(env, 'float32')
    env = ss.resize_v0(env, x_size=20, y_size=76)
    env = ss.flatten_v0(env)
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    env = ss.frame_stack_v1(env, 3)
    return env



env = env_creator()
env_name = "pistonball_v3"
register_env(env_name, lambda config: PettingZooEnv(env_creator()))

with open("/home/justinkterry/ray_results/pistonball_v3/PPO/PPO_pistonball_v3_53bee_00000_0_2021-02-01_04-24-08/params.pkl", "rb") as f:
    config = pickle.load(f)

ray.init()
PPOagent = PPOTrainer(env='pistonball_v3', config=config)
PPOagent.restore(checkpoint_path)


reward = 0
obs_list = []
i = 0
env.reset()

while True:
    for agent in env.agent_iter():
        observation, reward, done, info = env.last()
        reward += reward
        if done:
            action = None
        else:
            action, _, _ = PPOagent.get_policy("policy_0").compute_single_action(observation)

        env.step(action)
        i += 1
        if i % (len(env.possible_agents)+1) == 0:
            obs_list.append(np.transpose(env.render(mode='rgb_array'), axes=(1, 0, 2)))
    env.close()
    break


print(reward)
write_gif(obs_list, 'pistonball_160.gif', fps=15)
