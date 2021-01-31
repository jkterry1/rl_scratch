import ray
import os
import pickle
from ray.rllib.models import ModelCatalog
from run_rllib import MLPModelV2
from ray.tune.registry import register_trainable, register_env
from ray.rllib.agents.dqn import ApexTrainer
from ray.rllib.agents.ppo import PPOTrainer
from pettingzoo.butterfly import pistonball_v3
import supersuit as ss
from ray.rllib.env import PettingZooEnv
from array2gif import write_gif
from ray.tune.registry import get_trainable_cls

# path should end with checkpoint-<> data file
checkpoint_path = "~/ray_results/pistonball_v3/PPO/PPO_pistonball_v3_19368_00000_0_2021-01-30_20-45-33/checkpoint_100/checkpoint-100"

"""
def make_env_creator():
    def env_creator(args):
        env = pistonball_v3.env(n_pistons=10, local_ratio=0.2, time_penalty=-0.1, continuous=True, random_drop=True, random_rotate=True, ball_mass=0.75, ball_friction=0.3, ball_elasticity=1.5, max_cycles=900)
        env = ss.color_reduction_v0(env, mode='B')
        env = ss.dtype_v0(env, 'float32')
        env = ss.resize_v0(env, x_size=20, y_size=76)
        env = ss.flatten_v0(env)
        env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
        env = ss.frame_stack_v1(env, 3)
        return env
    return env_creator


ModelCatalog.register_custom_model("MLPModelV2", MLPModelV2)

env_creator = make_env_creator()

env_name = "pistonball_v3"

register_env(env_name, lambda config: PettingZooEnv(env_creator(config)))
"""

env = pistonball_v3.env(n_pistons=10, local_ratio=0.2, time_penalty=-0.1, continuous=True, random_drop=True, random_rotate=True, ball_mass=0.75, ball_friction=0.3, ball_elasticity=1.5, max_cycles=900)
env = ss.color_reduction_v0(env, mode='B')
env = ss.dtype_v0(env, 'float32')
env = ss.resize_v0(env, x_size=20, y_size=76)
env = ss.flatten_v0(env)
env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
env = ss.frame_stack_v1(env, 3)

# get the config file - params.pkl
config_path = os.path.dirname(checkpoint_path)
config_path = os.path.join(config_path, "../params.pkl")
with open(config_path, "rb") as f:
    config = pickle.load(f)
print('loaded')

ray.init()

cls = get_trainable_cls()
agent = cls(env=pistonball_v3, config=config)
agent.restore(checkpoint_path)

"""
Trainer = PPOTrainer
print ('pretrainer')
RLAgent = Trainer(env=pistonball_v3, config=config)
print('posttrainer')
RLAgent.restore(checkpoint_path)
"""

print('to playthrough')

"""
# init obs, action, reward
env.reset()
rewards, action_dict = {}, {}
for agent_id in env.agent_ids:
    assert isinstance(agent_id, int), "Error: agent_ids are not ints."
    # action_dict = dict(zip(env.agent_ids, [np.array([0,1,0]) for _ in range(len(env.agent_ids))])) # no action = [0,1,0]
    rewards[agent_id] = 0

totalReward = 0
"""
done = False

reward = 0
obs_list = []
iteration = 0

while not done:
    # action_dict = {}
    # compute_action does not cut it. Go to the policy directly
    for agent in env.agent_iter():
        # print("id {}, obs {}, rew {}".format(agent_id, observations[agent_id], rewards[agent_id]))
        observation, reward, done, info = env.last()
        reward += reward
        action, _, _ = agent.policy("policy_0").compute_single_action(observation)  # prev_action=action_dict[agent_id]
        # print(action)

        env.step(action)
        obs_list.append(env.render(mode='rgb_array'))
    #totalReward += sum(rewards.values())
    """
    done = any(list(dones.values()))
    print("iter:", iteration, sum(rewards.values()))
    iteration += 1
    """
print('playthrough over')
env.close()
print(reward)
write_gif(obs_list)
#print("done", done, totalReward)

# look into reward