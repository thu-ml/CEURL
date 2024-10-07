from collections import OrderedDict, deque
from typing import Any, NamedTuple
import os

import dm_env
import numpy as np
from dm_control import manipulation, suite
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import StepType, specs

import custom_dmc_tasks as cdmc
import gym
import pickle


class Gym2DMC(dm_env.Environment):
    def __init__(self, gym_env) -> None:
        gym_obs_space = gym_env.observation_space['rgb']
        # gym_obs_space = gym_env.observation_spec()[render_camera]
        self._observation_spec = specs.BoundedArray(
            shape=gym_obs_space.shape,
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name='observation'
        )
        self._action_spec = specs.BoundedArray(
            shape=gym_env.action_space.shape,
            dtype=np.float32,
            minimum=gym_env.action_space.low,
            maximum=gym_env.action_space.high,
            name='action'
        )
        self._gym_env = gym_env

    def step(self, action):
        obs, reward, done, info = self._gym_env.step(action)
        obs = obs['rgb']

        if done:
            step_type = StepType.LAST
            discount = 0.0
        else:
            step_type = StepType.MID
            discount = 1.0

        return dm_env.TimeStep(step_type=step_type,
                               reward=reward,
                               discount=discount,
                               observation=obs)

    def reset(self):
        obs = self._gym_env.reset()
        obs = obs['rgb']
        return dm_env.TimeStep(step_type=StepType.FIRST,
                               reward=None,
                               discount=None,
                               observation=obs)

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class FlattenJacoObservationWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env
        self._obs_spec = OrderedDict()
        wrapped_obs_spec = env.observation_spec().copy()
        if 'front_close' in wrapped_obs_spec:
            spec = wrapped_obs_spec['front_close']
            # drop batch dim
            self._obs_spec['pixels'] = specs.BoundedArray(shape=spec.shape[1:],
                                                          dtype=spec.dtype,
                                                          minimum=spec.minimum,
                                                          maximum=spec.maximum,
                                                          name='pixels')
            wrapped_obs_spec.pop('front_close')

        for key, spec in wrapped_obs_spec.items():
            assert spec.dtype == np.float64
            assert type(spec) == specs.Array
        dim = np.sum(
            np.fromiter((np.int(np.prod(spec.shape))
                         for spec in wrapped_obs_spec.values()), np.int32))

        self._obs_spec['observations'] = specs.Array(shape=(dim,),
                                                     dtype=np.float32,
                                                     name='observations')

    def _transform_observation(self, time_step):
        obs = OrderedDict()

        if 'front_close' in time_step.observation:
            pixels = time_step.observation['front_close']
            time_step.observation.pop('front_close')
            pixels = np.squeeze(pixels)
            obs['pixels'] = pixels

        features = []
        for feature in time_step.observation.values():
            features.append(feature.ravel())
        obs['observations'] = np.concatenate(features, axis=0)
        return time_step._replace(observation=obs)

    def reset(self):
        time_step = self._env.reset()
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key='pixels'):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        # print('wrapped_obs_spec', wrapped_obs_spec)
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(shape=np.concatenate(
            [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name='observation')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ObservationDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        self._dtype = dtype
        wrapped_obs_spec = env.observation_spec()['observations']
        self._obs_spec = specs.Array(wrapped_obs_spec.shape, dtype,
                                     'observation')

    def _transform_observation(self, time_step):
        obs = time_step.observation['observations'].astype(self._dtype)
        return time_step._replace(observation=obs)

    def reset(self):
        time_step = self._env.reset()
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class DreamerObsWrapper:
    def __init__(self, env):
        self._env = env
        self._ignored_keys = []
        self._action_dim = self._env.action_spec().shape[0]
        self._max_action_dim = self._action_dim
        self._spec_shape = self._env.action_spec().shape

    def set_max_act_dim(self, max_act_dim, max_act_shape):
        assert max_act_dim >= self._action_dim
        self._max_action_dim = max_act_dim
        self._spec_shape = max_act_shape

    @property
    def obs_space(self):
        spaces = {
            'observation': self._env.observation_spec(),
            'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            # np.bool -> bool
            # 'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
            # 'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
            # 'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
            'is_first': gym.spaces.Box(0, 1, (), dtype=bool),
            'is_last': gym.spaces.Box(0, 1, (), dtype=bool),
            'is_terminal': gym.spaces.Box(0, 1, (), dtype=bool),
        }
        return spaces

    @property
    def act_space(self):
        # always used for randomly sample action
        spec = self._env.action_spec()
        action = gym.spaces.Box((spec.minimum) * self._max_action_dim,
                                (spec.maximum) * self._max_action_dim,
                                shape=self._spec_shape,
                                dtype=np.float32)
        return {'action': action}

    def step(self, action):
        # assert np.isfinite(action['action']).all(), action['action']
        time_step = self._env.step(action[:self._action_dim])
        assert time_step.discount in (0, 1)
        obs = {
            'reward': time_step.reward,
            'is_first': False,
            'is_last': time_step.last(),
            'is_terminal': time_step.discount == 0,
            'observation': time_step.observation,
            'action': action,
            'discount': time_step.discount
        }
        return obs

    def reset(self):
        time_step = self._env.reset()
        obs = {
            'reward': 0.0,
            'is_first': True,
            'is_last': False,
            'is_terminal': False,
            'observation': time_step.observation,
            'action': np.zeros_like(self.act_space['action'].sample()),
            'discount': time_step.discount
        }
        return obs

    def __getattr__(self, name):
        if name == 'obs_space':
            return self.obs_space
        if name == 'act_space':
            return self.act_space
        return getattr(self._env, name)


class TimeLimit:

    def __init__(self, env, duration):
        self._env = env
        self._duration = duration
        self._step = None

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    def step(self, action):
        assert self._step is not None, 'Must reset environment.'
        ts, obs = self._env.step(action)
        self._step += 1
        if self._duration and self._step >= self._duration:
            ts = dm_env.TimeStep(dm_env.StepType.LAST, ts.reward, ts.discount, ts.observation)
            obs['is_last'] = True
            self._step = None
        return ts, obs

    def reset(self):
        self._step = 0
        return self._env.reset()

    def reset_with_task_id(self, task_id):
        self._step = 0
        return self._env.reset_with_task_id(task_id)


class NormalizeAction:

    def __init__(self, env, key='action'):
        self._env = env
        self._key = key
        space = env.act_space[key]
        self._mask = np.isfinite(space.low) & np.isfinite(space.high)
        self._low = np.where(self._mask, space.low, -1)
        self._high = np.where(self._mask, space.high, 1)

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def act_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        space = gym.spaces.Box(low, high, dtype=np.float32)
        return {**self._env.act_space, self._key: space}

    def step(self, action):
        orig = (action[self._key] + 1) / 2 * (self._high - self._low) + self._low
        orig = np.where(self._mask, orig, action[self._key])
        return self._env.step({**action, self._key: orig})


def _make_jaco(obs_type, domain, task, frame_stack, action_repeat, seed, img_size, ):
    env = cdmc.make_jaco(task, obs_type, seed, img_size, )
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = FlattenJacoObservationWrapper(env)
    env._size = (img_size, img_size)
    return env


def _make_dmc(obs_type, domain, task, frame_stack, action_repeat, seed, img_size,):
    visualize_reward = False
    if (domain, task) in suite.ALL_TASKS:
        env = suite.load(domain,
                         task,
                         task_kwargs=dict(random=seed),
                         environment_kwargs=dict(flat_observation=True),
                         visualize_reward=visualize_reward)
    else:
        env = cdmc.make(domain,
                        task,
                        task_kwargs=dict(random=seed),
                        environment_kwargs=dict(flat_observation=True),
                        visualize_reward=visualize_reward,)
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    if obs_type == 'pixels':
        # zoom in camera for quadruped
        camera_id = dict(quadruped=2).get(domain, 0)
        render_kwargs = dict(height=img_size, width=img_size, camera_id=camera_id)
        env = pixels.Wrapper(env,
                             pixels_only=True,
                             render_kwargs=render_kwargs)
        env._size = (img_size, img_size)
        env._camera = camera_id
    return env


def make(name, obs_type, frame_stack, action_repeat, seed, img_size=84, ):
    assert obs_type in ['states', 'pixels']
    # name:
    # walker_stand~mass~1.0
    # walker_stand~damping~1.0
    # walker2_stand~1.0
    # Panda_Door
    splited_name = name.split('~')
    name = splited_name[0]
    if len(splited_name) > 1:
        if splited_name[1] == 'mass':
            mass = splited_name[2]
            damping = None
            inertia = None
        elif splited_name[1] == 'damping':
            mass = None
            damping = splited_name[2]
            inertia = None
        # elif splited_name[1] == 'mass2':
        #     mass = splited_name[2]
        #     damping = None
        #     inertia = splited_name[2]
        else:
            raise ValueError("the name must be one of {mass, damping, mass2}")
    else:
        mass = None
        damping = None
        inertia = None
    domain, task = name.split('_', 1)
    domain = dict(cup='ball_in_cup', point='point_mass').get(domain, domain)

    if domain == 'jaco':
        make_fn = _make_jaco
        env = make_fn(obs_type, domain, task, frame_stack, action_repeat, seed, img_size, )
    else:
        make_fn = _make_dmc
        env = make_fn(obs_type, domain, task, frame_stack, action_repeat, seed, img_size, )

    if obs_type == 'pixels':
        env = FrameStackWrapper(env, frame_stack)
    else:
        env = ObservationDTypeWrapper(env, np.float32)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)

    env = ExtendedTimeStepWrapper(env)

    print('standard sum of mass, inertia, damping', sum(env._physics.model.body_mass),
          sum(env._physics.model.body_inertia), sum(env._physics.model.dof_damping))

    if mass is not None:
        env._physics.model.body_mass = float(mass) * env._physics.model.body_mass
    if inertia is not None:
        env._physics.model.body_inertia = float(inertia) * env._physics.model.body_inertia
    if damping is not None:
        env._physics.model.dof_damping = float(damping) * env._physics.model.dof_damping

    print('sum of mass, inertia, damping', sum(env._physics.model.body_mass),
            sum(env._physics.model.body_inertia), sum(env._physics.model.dof_damping))
    # env._physics.model.body_inertia

    return DreamerObsWrapper(env)
