import numpy as np
from pettingzoo.utils import wrappers
#from envs.jat.ma_quadx_hover_env import MAQuadXHoverEnv
from envs.jat.hover_v0 import hover_env as MAQuadXHoverEnv
from pettingzoo.test import parallel_api_test

start_pos = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0], [-1.0, 1.0, 1.0],[1.0, -1.0, 1.0], [-1.0, 1.0, 2.0], [-2.0, 2.0, 1.0], [-2.0, 1.0, 1.0] ])
start_orn = np.zeros_like(start_pos)

kwargs = dict(
            start_pos = start_pos,
            start_orn = start_orn
            )


env = MAQuadXHoverEnv(render_mode="human", **kwargs)
env = wrappers.OrderEnforcingWrapper(env)

parallel_api_test(env, num_cycles=1000)

env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample()

    env.step(action)
