from PyFlyt.pz_envs import MAQuadXHoverEnv
from stable_baselines3 import PPO
import numpy as np


model = PPO.load('C:/projects/pyflyt-hover/apps/jat/models/quadx_v0_20231206-125329.zip')

start_pos = np.array([[0.0, 0.0, 1.0]])
start_orn = np.zeros_like (start_pos)
seed=42

env = MAQuadXHoverEnv(render_mode="human", start_pos=start_pos, start_orn=start_orn)
observations, infos = env.reset()



while env.agents:
    # this is where you would insert your policy
    #actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    actions = {agent: model.predict(observations[agent], deterministic=True)[0] for agent in env.agents}
    
    observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()
