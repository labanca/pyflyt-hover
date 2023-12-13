from stable_baselines3 import PPO
from envs.labanca import hover_v0
import numpy as np


start_pos = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]])
start_orn = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
drone_type = ["quadx", "quadx"]

drone_options = []
drone_options.append(dict(control_hz=60))
drone_options.append(dict(control_hz=120))

env_kwargs = dict(start_pos=start_pos,
                        start_orn=start_orn,
                        render_mode="human",
                        drone_type=drone_type,
                        drone_options=drone_options)

#env = hover_v0.parallel_env(env_kwargs)

env = hover_v0.hover_env(**env_kwargs)

env.reset()
model = PPO.load("../../models/pyflyt_hover_20231202-184205.zip")

observation = {agent:env.observe(agent) for agent in env.last()}

while env.agents:
    #actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    actions = {agent: model.predict(observation[agent], deterministic=True)[0] for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)

    for agent in env.agents:
        if terminations[agent] == False:
            print(f'{agent} - {rewards[agent]} - term: {terminations[agent]}, trunc: {truncations[agent]}')
            print(f'{actions[agent]=}')
            print(f'{observations[agent]=}')

env.close()