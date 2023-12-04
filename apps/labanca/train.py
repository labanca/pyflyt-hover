from __future__ import annotations

import glob
import os
import time
import numpy as np
import supersuit as ss

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.utils import get_device
from datetime import datetime

from envs.labanca import hover_v0


def train_pyflyt_supersuit(
    env_fn, steps: int = 10_000, seed: int | None = 0, **env_kwargs
):
    # Train a single model to play as each agent in a cooperative Parallel environment

    env = env_fn.parallel_env(**env_kwargs)
    #env = ss.black_death_v3(env)

    device = get_device(device='cuda')
    env.reset(seed=seed)

    print(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=10, base_class="stable_baselines3")

    buffer_size = 10_000_000
    batch_size = 256
    n_envs = 1
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
    )

    model.learn(total_timesteps=steps)
    model_name = f"models/{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}"
    model.save(model_name)

    print("Model has been saved.")
    with open(f'{model_name}.txt', 'w') as file:
        # Write a string to the file
        file.write(f'{model.num_timesteps=}\n')
        start_datetime = datetime.fromtimestamp(model.start_time / 1e9)
        current_time = datetime.now()
        elapsed_time = current_time - start_datetime
        file.write(f'model.start_datetime={start_datetime}\n')
        file.write(f'completion_datetime={current_time}\n')
        file.write(f'elapsed_time={elapsed_time}\n')
        file.write(f'{model.policy_kwargs=}\n')
        file.write(f'{model.device=}\n')
        file.write(f'{model.learning_rate=}\n')
        file.write(f'{model.policy=}\n')

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()


def eval(env_fn, num_games: int = 100, render_mode: str | None = None, **env_kwargs):
    # Evaluate a trained agent vs a random agent
    env = env_fn.hover_env(render_mode=render_mode, **env_kwargs)

    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    try:
        latest_policy = max(
            glob.glob(f"models/{env.metadata['name']}*.zip"), key=os.path.getctime

        )
        print(f'{latest_policy=}')

    except ValueError:
        print("Policy not found.")
        exit(0)

    model = PPO.load(latest_policy)

    rewards = {agent: 0 for agent in env.possible_agents}

    # Note: We train using the Parallel API but evaluate using the AEC API
    # SB3 models are designed for single-agent settings, we get around this by using he same model for every agent
    for i in range(num_games):
        env.reset(seed=i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            for agent in env.agents:
                rewards[agent] += env.rewards[agent]
            if termination or truncation:
                break
            else:
                act = model.predict(obs, deterministic=True)[0]

            env.step(act)
            print(f'{env.env.aviary.elapsed_time=}{env.terminations=}{env.truncations=}')
            print(f'{act}')
            print(reward)
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    print("Rewards: ", rewards)
    print(f"Avg reward: {avg_reward}")
    return avg_reward


if __name__ == "__main__":
    env_fn = hover_v0

    # mode 0: vp, vq, vr, T: angular velocities + Thrust
    start_pos = np.array([[-1.0, 1.0, 2.0]])
    #start_pos = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]])
    #start_pos = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0], [-1.0, 1.0, 1.0],[1.0, -1.0, 1.0], [-1.0, 1.0, 2.0], [-2.0, 2.0, 1.0], [-2.0, 1.0, 1.0] ])
    start_orn = np.zeros_like (start_pos)
    drone_type = ["quadx"] * start_pos.shape[0]

    env_kwargs = {
        'start_pos': start_pos,
        'start_orn': start_orn,
        #'render_mode': None,
    }

    steps = 2000_000

    # Train a model (takes ~3 minutes on GPU)
    #train_pyflyt_supersuit(env_fn, steps=steps, seed=42,  **env_kwargs)

    # Evaluate 10 games (average reward should be positive but can vary significantly)
    #eval(env_fn, num_games=10, **env_kwargs)

    # Watch 2 games
    eval(env_fn, num_games=10, render_mode="human", **env_kwargs)