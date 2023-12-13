from __future__ import annotations

import time
import numpy as np
import supersuit as ss
import torch
from PyFlyt.core import Aviary
from gymnasium.utils import EzPickle

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from PyFlyt.pz_envs import MAQuadXHoverEnv


class Hover(MAQuadXHoverEnv, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "quadx_v0",
        "is_parallelizable": True,
        "render_fps": 30,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, render_mode=kwargs['render_mode'], start_pos= kwargs['start_pos'], start_orn= kwargs['start_orn'])
        EzPickle.__init__(self, *args, **kwargs)
        # this made the training work, but I believe this is not right
        #self.env = Aviary(**env_kwargs)

start_pos = np.array([[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [-1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
start_orn = np.zeros_like (start_pos)
seed=42
steps = 1_000_000

env_kwargs = {
    'start_pos': start_pos,
    'start_orn': start_orn,
    'drone_type': 'quadx'
}

# Train a single model to play as each agent in a cooperative Parallel environment
env_fn = Hover(render_mode=None, **env_kwargs)
env = env_fn
env = ss.black_death_v3(env)
env.reset(seed=seed)
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 1, num_cpus=10, base_class="stable_baselines3")

# train params
device = torch.device(device='cpu')
batch_size = 256
lr = 0.0007
nn_t = [128, 128, 128]
policy_kwargs = dict(
    #normalize_images=False,
    net_arch=dict(pi=nn_t, vf=nn_t)
)

model = PPO(
    MlpPolicy,
    env,
    verbose=1,
    learning_rate=lr,
    batch_size=batch_size,
    policy_kwargs=policy_kwargs,
    device=device
)

print(f"Starting training on {str(env.metadata['name'])}.")
model.learn(total_timesteps=steps)
model_name = f"models/{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}"
model.save(model_name)