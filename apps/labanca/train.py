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
#from envs.jat import hover_v0

def train_pyflyt_supersuit(
    env_fn, steps: int = 10_000, seed: int | None = 0, **env_kwargs
):
    # Train a single model to play as each agent in a cooperative Parallel environment

    env = env_fn.parallel_env(**env_kwargs)
    #env = ss.black_death_v3(env)

    device = get_device(device='cpu')
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
        device=device
    )

    model.learn(total_timesteps=steps)
    model_name = f"models/{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}"
    model.save(model_name)

    print("Model has been saved.")
    with open(f'{model_name}.txt', 'w') as file:
        # Write a string to the file
        file.write(f'{__file__=}\n')
        file.write(f'{model.num_timesteps=}\n')
        file.write(f'{device=}\n')
        file.write(f'{start_pos=}\n')
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
        f"/nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    # try:
    #     latest_policy = max(
    #         glob.glob(f"models/{env.metadata['name']}*.zip"), key=os.path.getctime
    #
    #     )
    #     print(f'{latest_policy=}')

    # except ValueError:
    #     print("Policy not found.")
    #     exit(0)

    #model = PPO.load(latest_policy)
    model = PPO.load('C:/projects/pyflyt-hover/apps/labanca/models/pyflyt_hover_20231204-054100.zip')

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
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    print("Rewards: ", rewards)
    print(f"Avg reward: {avg_reward}")
    return avg_reward

def eval_positions(env_fn, num_games: int = 100, render_mode: str | None = None, **env_kwargs):
    # Evaluate a trained agent vs a random agent
    import pandas as pd

    env = env_fn.hover_env(render_mode=render_mode, **env_kwargs)

    print(
        f"/nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    # try:
    #     latest_policy = max(
    #         glob.glob(f"models/{env.metadata['name']}*.zip"), key=os.path.getctime
    #
    #     )
    #     print(f'{latest_policy=}')

    # except ValueError:
    #     print("Policy not found.")
    #     exit(0)
    model_filename = 'C:/projects/pyflyt-hover/apps/labanca/models/pyflyt_hover_20231204-054100.zip'
    #model = PPO.load(latest_policy)
    model = PPO.load(model_filename)

    rewards = {agent: 0 for agent in env.possible_agents}

    # Note: We train using the Parallel API but evaluate using the AEC API
    # SB3 models are designed for single-agent settings, we get around this by using he same model for every agent


    results = [['i', 'env.env.aviary.elapsed_time', 'agent', 'act', 'rewards', 'termination', 'truncation', 'info', 'model', 'num_agents', 'posx', 'posy', 'posz', 'all_start_pos', 'agent_start_pos', 'obs' ]]

    for i in range(num_games):

        env.reset(seed=i)

        for agent in env.agent_iter():

            agent_id = env.env.agent_name_mapping
            obs, reward, termination, truncation, info = env.last()


            for agent in env.agents:
                rewards[agent] += env.rewards[agent]

            if termination or truncation:
                results.append([i, env.env.aviary.elapsed_time, agent, act, rewards, termination, truncation, info, model_filename, env.num_agents, obs[9],obs[10], obs[11], env.env.start_pos, env.env.start_pos[env.env.agent_name_mapping[agent]], obs])
                break
            else:
                act = model.predict(obs, deterministic=True)[0]

            env.step(act)


    env.close()

    df = pd.DataFrame(results[1:], columns=results[0])
    #df.reset_index(drop=True, inplace=True)
    df.to_csv(f"evals/{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}.csv", index=False)


    avg_reward = sum(rewards.values()) / len(rewards.values())
    print("Rewards: ", rewards)
    print(f"Avg reward: {avg_reward}")
    return avg_reward


def generate_initial_positions(num_drones, min_distance, spawn_radius, center):
    start_pos = np.empty((num_drones, 3))
    min_z = 1

    for i in range(num_drones):
        while True:
            # Generate random coordinates within the spawn area centered at 'center'
            x = np.random.uniform(center[0] - spawn_radius, center[0] + spawn_radius)
            y = np.random.uniform(center[1] - spawn_radius, center[1] + spawn_radius)
            z = np.random.uniform(max(center[2], min_z), center[2] + spawn_radius)  # Ensure z-axis is within range

            # Check if the minimum distance condition is met with existing drones
            if i == 0 or np.min(np.linalg.norm(start_pos[:i] - np.array([x, y, z]), axis=1)) >= min_distance:
                start_pos[i] = [x, y, z]
                break

    return start_pos

if __name__ == "__main__":
    env_fn = hover_v0

    spawn_settings = dict(
        num_drones = 3,
        min_distance = 1.0,
        spawn_radius = 2.0,
        center = (0, 0, 0),
    )
    # mode 0: vp, vq, vr, T: angular velocities + Thrust

    start_pos = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]])
    #start_pos = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0], [-1.0, 1.0, 1.0]])
    #wstart_pos = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0], [-1.0, 1.0, 1.0], [3.0, -1.0, 1.0]])
    #start_pos = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0], [-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [-1.0, 1.0, 2.0]])
    #start_pos = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0], [-1.0, 1.0, 1.0],[1.0, -1.0, 1.0], [-1.0, 1.0, 2.0], [-2.0, 2.0, 1.0], [-2.0, 1.0, 1.0], [-3.0, 1.0, 1.0] ])
    start_orn = np.zeros_like (start_pos)
    drone_type = ["quadx"] * start_pos.shape[0]

    env_kwargs = {
        'start_pos': start_pos,
        'start_orn': start_orn,
        'spawn_settings': spawn_settings,
    }

    steps = 10_000_000

    # Train a model (takes ~3 minutes on GPU)
    #train_pyflyt_supersuit(env_fn, steps=steps, seed=42,  **env_kwargs)

    # Evaluate 10 games (average reward should be positive but can vary significantly)
    #eval(env_fn, num_games=10, **env_kwargs)

    # Watch games
    #eval(env_fn, num_games=1, render_mode="human", **env_kwargs)

    # eval spawn positions
    eval_positions(env_fn, num_games=50000, render_mode=None , **env_kwargs)