from custom_dmc_tasks import walker
from custom_dmc_tasks import quadruped
from custom_dmc_tasks import jaco


def make(domain, task,
         task_kwargs=None,
         environment_kwargs=None,
         visualize_reward=False,
         mass=1.0):
    
    if domain == 'walker':
        return walker.make(task,
                           task_kwargs=task_kwargs,
                           environment_kwargs=environment_kwargs,
                           visualize_reward=visualize_reward)
    elif domain == 'quadruped':
        return quadruped.make(task,
                              task_kwargs=task_kwargs,
                              environment_kwargs=environment_kwargs,
                              visualize_reward=visualize_reward)
    else:
        raise f'{task} not found'


def make_jaco(task, obs_type, seed, img_size,):
    return jaco.make(task, obs_type, seed, img_size,)