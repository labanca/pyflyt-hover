from gymnasium.utils import EzPickle
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers

from pettingzoo.utils.conversions import parallel_wrapper_fn
from envs.jat.ma_quadx_hover_env import MAQuadXHoverEnv



def hover_env(render_mode=None, **kwargs):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    env = raw_hover_env(render_mode=render_mode, **kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

parallel_env = parallel_wrapper_fn(hover_env)

class raw_hover_env(MAQuadXHoverEnv, AECEnv, EzPickle):
    def __init__(self,
                 *args,
                 **kwargs
                 ):
        MAQuadXHoverEnv.__init__(self, *args, **kwargs)
        EzPickle.__init__(self, *args, **kwargs)


def api_test():
    from pettingzoo.test import parallel_api_test
    import hover_v0
    import numpy as np

    start_pos = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]])
    start_orn = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    drone_type = ["quadx", "quadx"]

    env = hover_v0.parallel_env(
        start_pos=start_pos,
        start_orn=start_orn,
        render_mode=None,
        #drone_type=drone_type
    )
    parallel_api_test(env, num_cycles=1000)


if __name__ == "__main__":

    api_test()

    env = hover_env(render_mode="human")
    env.reset(seed=42)

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        print(observation)

        if termination or truncation:
            action = None
        else:
            # this is where you would insert your policy
            action = env.action_space(agent).sample()
            #action = np.array([0.0, 0.0, 0.0, 0.0])

        env.step(action)
    env.close()

