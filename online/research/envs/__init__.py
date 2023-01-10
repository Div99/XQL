# Register environment classes here
from .empty import Empty

# If we want to register environments in gym.
# These will be loaded when we import the research package.
from gym.envs import register

# Register the DM Control environments.
from dm_control import suite

# Custom DM Control domains can be registered as follows:
# from . import <custom dm_env module>
# assert hasattr(<custom dm_env module>, 'SUITE')
# suite._DOMAINS['<custom dm_env module>'] = <custom dm_env module>

# Register all of the DM control tasks
for domain_name, task_name in suite._get_tasks(tag=None):
    # Import state domains
    ID = f'{domain_name.capitalize()}{task_name.capitalize()}-v0'
    register(id=ID, 
             entry_point='research.envs.dm_control:DMControlEnv', 
             kwargs={'domain_name': domain_name, 
                     'task_name': task_name,
                     'action_minimum': -1.0,
                     'action_maximum': 1.0,
                     'action_repeat': 1,
                     'from_pixels': False,
                     'flatten': True,
                     'stack': 1}, 
             )

    # Import vision domains as specified in DRQ-v2
    ID = f'{domain_name.capitalize()}{task_name.capitalize()}-vision-v0'
    camera_id = dict(quadruped=2).get(domain_name, 0)
    register(id=ID, 
             entry_point='research.envs.dm_control:DMControlEnv', 
             kwargs={'domain_name': domain_name, 
                     'task_name': task_name,
                     'action_repeat': 2,
                     'action_minimum': -1.0,
                     'action_maximum': 1.0,
                     'from_pixels': True,
                     'height': 84,
                     'width': 84,
                     'camera_id': camera_id,
                     'flatten': False,
                     'stack': 3}, 
             )

# Cleanup extra imports
del suite
del register
