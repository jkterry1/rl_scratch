# Ben's original inversion


env = pong_v1.env()
def invert_agent_indication(obs, agent):
    obs2 = obs if agent == env.possible_agents[0] else 255-obs
    return np.concatenate([obs, obs2],axis=0)
env = observation_lambda_v0(env, invert_agent_indication)


# Ben's new inversion

def InvertColorPlusAgentIndicator(env):
    def modify_obs(obs, agent):
        num_agents = len(env.possible_agents)
        agent_idx = env.possible_agents.index(agent)
        if num_agents == 2:
            if agent_idx == 1:
                rotated_obs = 255 - obs
            else:
                rotated_obs = obs
        elif num_agents == 4:
            rotated_obs = (255*agent_idx)//4 + obs

        indicator = np.zeros((2, )+obs.shape[1:],dtype="uint8")
        indicator[0] = 255 * agent_idx % 2
        indicator[1] = 255 * ((agent_idx+1) // 2) % 2
        return np.concatenate([obs, rotated_obs, indicator], axis=0)
    env = ss.observation_lambda_v0(env, modify_obs)
    env = ss.pad_observations_v0(env)
    return env


class InvertColorAgentIndicator(ObservationWrapper):
    def _check_wrapper_params(self):
        assert self.observation_spaces[self.possible_agents[0]].high.dtype == np.dtype('uint8')
        return

    def _modify_spaces(self):
        return

    def _modify_observation(self, agent, observation):
        max_num_agents = len(self.possible_agents)
        if max_num_agents == 2:
            if agent == self.possible_agents[1]:
                return self.observation_spaces[agent].high - observation
            else:
                return observation
        elif max_num_agents == 4:
            if agent == self.possible_agents:
                return np.uint8(255//4)+observation

# Ryan's mess https://gist.github.com/benblack769/cbf4c0a674ad24d0e095263a0b553726#file-new_param_sharing_rainbow-py-L188