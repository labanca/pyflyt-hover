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
