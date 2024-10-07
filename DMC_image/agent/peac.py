import torch.nn as nn
import torch
from torch.nn import functional as F
import torch.distributions as D

import utils
import agent.dreamer_utils as common
from collections import OrderedDict
import numpy as np
import math

# from agent.dreamer import ActorCritic


def stop_gradient(x):
    return x.detach()


def get_feat_ac(seq):
    return torch.cat([seq['feat'], seq['context']], dim=-1)


Module = nn.Module


class TaskModel(nn.Module):
    def __init__(self, obs_dim, task_dim, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, task_dim))

        self.apply(utils.weight_init)

    def forward(self, obs):
        state_feat = self.model(obs)
        return state_feat


class PEACAgent(Module):

    def __init__(self,
                 name, cfg, obs_space, act_spec, task_number, **kwargs):
        super().__init__()
        self.name = name
        self.cfg = cfg
        self.cfg.update(**kwargs)
        self.obs_space = obs_space
        self.act_spec = act_spec
        self.tfstep = None
        self._use_amp = (cfg.precision == 16)
        self.device = cfg.device
        self.act_dim = act_spec.shape[0]

        self.task_number = task_number

        self.wm = WorldModel(cfg, obs_space, self.act_dim, self.tfstep,
                             task_number=self.task_number)
        self.reward_free = True
        self._task_behavior = ContextActorCritic(cfg, self.act_spec, self.tfstep,
                                                 context_dim=self.task_number)
        self.to(cfg.device)
        self.requires_grad_(requires_grad=False)

    def act(self, obs, meta, step, eval_mode, state):
        obs = {k: torch.as_tensor(np.copy(v), device=self.device).unsqueeze(0) for k, v in obs.items()}
        if state is None:
            latent = self.wm.rssm.initial(len(obs['reward']))
            action = torch.zeros((len(obs['reward']),) + self.act_spec.shape, device=self.device)
        else:
            latent, action = state
        embed = self.wm.encoder(self.wm.preprocess(obs))
        should_sample = (not eval_mode) or (not self.cfg.eval_state_mean)
        latent, _ = self.wm.rssm.obs_step(latent, action, embed, obs['is_first'], should_sample)
        feat = self.wm.rssm.get_feat(latent)
        context = F.softmax(self.wm.task_model(feat), dim=-1)
        inp = torch.cat([feat, context], dim=-1)
        if eval_mode:
            actor = self._task_behavior.actor(inp)
            action = actor.mean
        else:
            actor = self._task_behavior.actor(inp)
            action = actor.sample()
        new_state = (latent, action)
        return action.cpu().numpy()[0], new_state

    def update_wm(self, data, step):
        metrics = {}
        state, outputs, mets = self.wm.update(data, state=None)
        outputs['is_terminal'] = data['is_terminal']
        metrics.update(mets)
        return state, outputs, metrics

    def update(self, data, step):
        state, outputs, metrics = self.update_wm(data, step)

        start = outputs['post']

        if self.cfg.reward_type == 4:
            start_prior = outputs['prior']
            with torch.no_grad():
                start['prior_feat'] = self.wm.rssm.get_feat(start_prior)

        # Don't train the policy/value if just using MPC
        if getattr(self.cfg, "mpc", False) and (not self.cfg.mpc_opt.use_value):
            return state, metrics
        start['embodiment_id'] = data['embodiment_id']
        start['context'] = outputs['context']
        # print('hhh', outputs['post'].keys())
        # print('iii', outputs['prior'].keys())
        # start['prior_feat'] = outputs['prior']['feat']
        start = {k: stop_gradient(v) for k, v in start.items()}
        # print('data', data.keys())
        if self.reward_free:
            # print('we use calculated reward')
            reward_fn = lambda seq: self.compute_task_reward(seq, step)
        else:
            reward_fn = lambda seq: self.wm.heads['reward'](seq['feat']).mean  # .mode()
        metrics.update(self._task_behavior.update(
            self.wm, start, data['is_terminal'], reward_fn))
        return state, metrics

    def report(self, data):
        report = {}
        data = self.wm.preprocess(data)
        for key in self.wm.heads['decoder'].cnn_keys:
            name = key.replace('/', '_')
            report[f'openl_{name}'] = self.wm.video_pred(data, key)
        return report

    def get_meta_specs(self):
        return tuple()

    def init_meta(self):
        return OrderedDict()

    def update_meta(self, meta, global_step, time_step, finetune=False):
        return meta

    def init_from(self, other):
        # TODO: update task model?
        init_task = self.cfg.get('init_task', 1.0)
        init_critic = self.cfg.get('init_critic', False)
        init_actor = self.cfg.get('init_actor', True)

        # copy parameters over
        print(f"Copying the pretrained world model")
        utils.hard_update_params(other.wm.rssm, self.wm.rssm)
        utils.hard_update_params(other.wm.encoder, self.wm.encoder)
        utils.hard_update_params(other.wm.heads['decoder'], self.wm.heads['decoder'])

        if init_task > 0.0:
            print(f"Copying the task model")
            utils.hard_update_params(other.wm.task_model, self.wm.task_model)

        if init_actor:
            print(f"Copying the pretrained actor")
            utils.hard_update_params(other._task_behavior.actor, self._task_behavior.actor)

        if init_critic:
            print(f"Copying the pretrained critic")
            utils.hard_update_params(other._task_behavior.critic, self._task_behavior.critic)
            if self.cfg.slow_target:
                utils.hard_update_params(other._task_behavior._target_critic, self._task_behavior._target_critic)

    @torch.no_grad()
    def estimate_value(self, start, actions, horizon):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}
        start['feat'] = self.wm.rssm.get_feat(start)
        start['action'] = torch.zeros_like(actions[0], device=self.device)
        seq = {k: [v] for k, v in start.items()}
        for t in range(horizon):
            action = actions[t]
            state = self.wm.rssm.img_step({k: v[-1] for k, v in seq.items()}, action)
            feat = self.wm.rssm.get_feat(state)
            for key, value in {**state, 'action': action, 'feat': feat}.items():
                seq[key].append(value)

        seq = {k: torch.stack(v, 0) for k, v in seq.items()}
        reward = self.wm.heads['reward'](seq['feat']).mean
        if self.cfg.mpc_opt.use_value:
            value = self._task_behavior._target_critic(get_feat_ac(seq)).mean
        else:
            value = torch.zeros_like(reward, device=self.device)
        disc = self.cfg.discount * torch.ones(list(seq['feat'].shape[:-1]) + [1], device=self.device)

        lambda_ret = common.lambda_return(
            reward[:-1], value[:-1], disc[:-1],
            bootstrap=value[-1],
            lambda_=self.cfg.discount_lambda,
            axis=0)

        # First step is lost because the reward is from the start state
        return lambda_ret[1]

    @torch.no_grad()
    def plan(self, obs, meta, step, eval_mode, state, t0=True):
        """
        Plan next action using Dyna-MPC.
        We thank the authors of TD-MPC (https://github.com/nicklashansen/tdmpc), to provide a good reference for implementing our planning strategy.
        """

        # Get Dreamer's state and features
        obs = {k: torch.as_tensor(np.copy(v), device=self.device).unsqueeze(0) for k, v in obs.items()}
        if state is None:
            latent = self.wm.rssm.initial(len(obs['reward']))
            action = torch.zeros((len(obs['reward']),) + self.act_spec.shape, device=self.device)
        else:
            latent, action = state
        embed = self.wm.encoder(self.wm.preprocess(obs))
        should_sample = (not eval_mode) or (not self.cfg.eval_state_mean)
        post, prior = self.wm.rssm.obs_step(latent, action, embed, obs['is_first'], should_sample)
        feat = self.wm.rssm.get_feat(post)

        # Sample policy trajectories
        num_pi_trajs = int(self.cfg.mpc_opt.mixture_coef * self.cfg.mpc_opt.num_samples)
        if num_pi_trajs > 0:
            start = {k: v.repeat(num_pi_trajs, *list([1] * len(v.shape))) for k, v in post.items()}
            seq = self.wm.imagine(self._task_behavior.actor, start, None, self.cfg.mpc_opt.horizon)
            pi_actions = seq['action'][1:]

        # Initialize state and parameters
        start = {k: v.repeat(self.cfg.mpc_opt.num_samples + num_pi_trajs, *list([1] * len(v.shape))) for k, v in
                 post.items()}
        mean = torch.zeros(self.cfg.mpc_opt.horizon, self.act_dim, device=self.device)
        std = 2 * torch.ones(self.cfg.mpc_opt.horizon, self.act_dim, device=self.device)
        if not t0 and hasattr(self, '_prev_mean'):
            mean[:-1] = self._prev_mean[1:]

        # Iterate CEM
        for i in range(self.cfg.mpc_opt.iterations):
            actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * \
                                  torch.randn(self.cfg.mpc_opt.horizon, self.cfg.mpc_opt.num_samples, self.act_dim,
                                              device=std.device), -1, 1)
            if num_pi_trajs > 0:
                actions = torch.cat([actions, pi_actions], dim=1)

            # Compute elite actions
            value = self.estimate_value(start, actions, self.cfg.mpc_opt.horizon)
            elite_idxs = torch.topk(value.squeeze(1), self.cfg.mpc_opt.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            # Update parameters
            max_value = elite_value.max(0)[0]
            score = torch.exp(self.cfg.mpc_opt.temperature * (elite_value - max_value))
            score /= score.sum(0)
            _mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
            _std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - _mean.unsqueeze(1)) ** 2, dim=1) / (
                        score.sum(0) + 1e-9))
            _std = _std.clamp_(self.cfg.mpc_opt.min_std, 2)
            mean, std = self.cfg.mpc_opt.momentum * mean + (1 - self.cfg.mpc_opt.momentum) * _mean, _std

        # Outputs
        score = score.squeeze(1).cpu().numpy()
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        self._prev_mean = mean
        mean, std = actions[0], _std[0]
        a = mean
        if not eval_mode:
            a += std * torch.randn(self.act_dim, device=std.device)
        new_state = (post, a.unsqueeze(0))
        return a.cpu().numpy(), new_state

    def finetune_mode(self):
        self.is_ft = True
        self.reward_free = False

        # self._task_behavior.rewnorm = common.StreamNorm(**{"momentum": 1.00, "scale": 1.0, "eps": 1e-8},
        #                                                 device=self.device)
        # self.cfg.actor_ent = 1e-4
        # self.cfg.skill_actor_ent = 1e-4

    def compute_task_reward(self, seq, step):
        # print('we use calculated reward')
        # print(seq.keys())
        B, T, _ = seq['feat'].shape
        task_pred = self.wm.task_model(seq['feat'])
        task_truth = seq['embodiment_id'].repeat(B, 1, 1).to(dtype=torch.int64)

        # calculate the task model prediction loss v2
        if self.cfg.reward_type == 1:
            task_pred = F.softmax(task_pred, dim=2)
            intr_rew = torch.zeros(task_pred.shape, device=self.device)  # 16, 2500, task_number
            intr_rew = intr_rew.reshape(B*T, -1)
            intr_rew[torch.arange(B*T), task_truth.reshape(-1)] = 1.0
            intr_rew = intr_rew.reshape(B, T, -1)
            intr_rew = torch.sum(torch.square(intr_rew - task_pred), dim=2, keepdim=True)
        # calculate the task model predict prob
        elif self.cfg.reward_type == 2:
            task_pred = F.log_softmax(task_pred, dim=2)
            intr_rew = task_pred.reshape(B*T, -1)[torch.arange(B*T), task_truth.reshape(-1)]
            intr_rew = -intr_rew.reshape(B, T, 1)
        # calculate the task model predict prob - entropy
        elif self.cfg.reward_type == 3:
            task_pred = F.log_softmax(task_pred, dim=2)
            entropy = task_pred.sum(dim=2, keepdim=True) / task_pred.shape[2] # B, T, 1
            intr_rew = task_pred.reshape(B*T, -1)[torch.arange(B*T), task_truth.reshape(-1)]
            intr_rew = -(intr_rew.reshape(B, T, 1) - entropy)
        elif self.cfg.reward_type == 4:
            # kld = D.kl_divergence
            task_pred = F.softmax(task_pred, dim=2)
            task_prior_pred = F.softmax(self.wm.task_model(seq['prior_feat'].
                                                           repeat(B, 1, 1)), dim=2)
            # task_rew = torch.sum(torch.square(task_pred - task_prior_pred), dim=2, keepdim=True)
            task_pred_dist = D.Categorical(task_pred)
            task_prior_pred_dist = D.Categorical(task_prior_pred)
            intr_rew = D.kl_divergence(task_pred_dist, task_prior_pred_dist).reshape(B, T, 1)
        else:
            raise Exception('Current reward type is {}, which is not supported'.
                            format(self.cfg.agent.reward_type))

        # print(intr_rew.shape) # 16, 2500, 1
        return intr_rew


class WorldModel(Module):
    def __init__(self, config, obs_space, act_dim, tfstep, task_number=1):
        super().__init__()
        shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
        self.cfg = config
        self.device = config.device
        self.tfstep = tfstep
        self.encoder = common.Encoder(shapes, **config.encoder)
        # Computing embed dim
        with torch.no_grad():
            zeros = {k: torch.zeros((1,) + v) for k, v in shapes.items()}
            outs = self.encoder(zeros)
            embed_dim = outs.shape[1]
        self.embed_dim = embed_dim
        self.rssm = common.EnsembleRSSM(**config.rssm, action_dim=act_dim, embed_dim=embed_dim, device=self.device)
        self.heads = {}
        self._use_amp = (config.precision == 16)
        inp_size = config.rssm.deter
        if config.rssm.discrete:
            inp_size += config.rssm.stoch * config.rssm.discrete
        else:
            inp_size += config.rssm.stoch
        self.inp_size = inp_size
        self.heads['decoder'] = common.Decoder(shapes, **config.decoder, embed_dim=inp_size)
        self.heads['reward'] = common.MLP(inp_size, (1,), **config.reward_head)

        self.task_number = task_number
        if self.task_number > 1:
            # self.heads['embodiment_id'] = common.MLP(inp_size, (task_number,), **config.task_head)
            self.task_model = TaskModel(inp_size, task_number, config.task_head.units)
        else:
            self.task_model = None
        if config.pred_discount:
            self.heads['discount'] = common.MLP(inp_size, (1,), **config.discount_head)
        for name in config.grad_heads:
            assert name in self.heads, name
        self.grad_heads = config.grad_heads
        self.heads = nn.ModuleDict(self.heads)
        self.model_opt = common.Optimizer('model', self.parameters(), **config.model_opt, use_amp=self._use_amp)

    def update(self, data, state=None):
        with common.RequiresGrad(self):
            with torch.cuda.amp.autocast(enabled=self._use_amp):
                model_loss, state, outputs, metrics = self.loss(data, state)
            metrics.update(self.model_opt(model_loss, self.parameters()))
        return state, outputs, metrics

    def loss(self, data, state=None):
        data = self.preprocess(data)
        embed = self.encoder(data)
        post, prior = self.rssm.observe(
            embed, data['action'], data['is_first'], state)
        kl_loss, kl_value = self.rssm.kl_loss(post, prior, **self.cfg.kl)
        assert len(kl_loss.shape) == 0 or (len(kl_loss.shape) == 1 and kl_loss.shape[0] == 1), kl_loss.shape
        likes = {}
        losses = {'kl': kl_loss}
        feat = self.rssm.get_feat(post)
        for name, head in self.heads.items():
            # print('name', name)
            # print('input', feat.shape)
            grad_head = (name in self.grad_heads)
            inp = feat if grad_head else stop_gradient(feat)
            out = head(inp)
            dists = out if isinstance(out, dict) else {name: out}
            for key, dist in dists.items():
                # print('key:', key)
                # print('output', data[key].shape)
                # print('dist', dist)
                like = dist.log_prob(data[key])
                likes[key] = like
                losses[key] = -like.mean()

        context = None
        if self.task_number > 1:
            # print('task task task')
            out = self.task_model(feat)
            B, T, _ = out.shape
            out = out.reshape(B*T, -1)
            embodiment_id_key = data['embodiment_id'].reshape(B*T).to(torch.int64)
            # print('oo', out.shape)
            # print('key', data['embodiment_id'].shape)
            # print(embodiment_id_key)
            losses['embodiment_id'] = F.cross_entropy(out, embodiment_id_key)
            context = F.softmax(out.reshape(B, T, -1), dim=2)

        model_loss = sum(
            self.cfg.loss_scales.get(k, 1.0) * v for k, v in losses.items())
        outs = dict(
            embed=embed, feat=feat, post=post,
            prior=prior, likes=likes, kl=kl_value, context=context)
        metrics = {f'{name}_loss': value for name, value in losses.items()}
        metrics['model_kl'] = kl_value.mean()
        metrics['prior_ent'] = self.rssm.get_dist(prior).entropy().mean()
        metrics['post_ent'] = self.rssm.get_dist(post).entropy().mean()
        last_state = {k: v[:, -1] for k, v in post.items()}
        # last_state.keys(): stoch, deter, logits
        return model_loss, last_state, outs, metrics

    def imagine(self, policy, start, is_terminal, horizon, task_cond=None, eval_policy=False):
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}
        start['feat'] = self.rssm.get_feat(start)
        inp = start['feat'] if task_cond is None else torch.cat([start['feat'], task_cond], dim=-1)
        start['action'] = torch.zeros_like(policy(inp).mean, device=self.device)  # .mode())
        seq = {k: [v] for k, v in start.items()}
        if task_cond is not None:
            seq['task'] = [task_cond]
        for _ in range(horizon):
            inp = seq['feat'][-1] if task_cond is None else torch.cat([seq['feat'][-1], task_cond], dim=-1)
            action = policy(stop_gradient(inp)).sample() if not eval_policy else policy(stop_gradient(inp)).mean
            state = self.rssm.img_step({k: v[-1] for k, v in seq.items()}, action)
            feat = self.rssm.get_feat(state)
            for key, value in {**state, 'action': action, 'feat': feat}.items():
                seq[key].append(value)
            if task_cond is not None:
                seq['task'].append(task_cond)
        # shape will be (T, B, *DIMS)
        seq = {k: torch.stack(v, 0) for k, v in seq.items()}
        if 'discount' in self.heads:
            disc = self.heads['discount'](seq['feat']).mean()
            if is_terminal is not None:
                # Override discount prediction for the first step with the true
                # discount factor from the replay buffer.
                true_first = 1.0 - flatten(is_terminal)
                true_first *= self.cfg.discount
                disc = torch.cat([true_first[None], disc[1:]], 0)
        else:
            disc = self.cfg.discount * torch.ones(list(seq['feat'].shape[:-1]) + [1], device=self.device)
        seq['discount'] = disc
        # Shift discount factors because they imply whether the following state
        # will be valid, not whether the current state is valid.
        seq['weight'] = torch.cumprod(
            torch.cat([torch.ones_like(disc[:1], device=self.device), disc[:-1]], 0), 0)
        return seq

    def preprocess(self, obs):
        obs = obs.copy()
        for key, value in obs.items():
            if key.startswith('log_'):
                continue
            if value.dtype in [np.uint8, torch.uint8]:
                value = value / 255.0 - 0.5
            obs[key] = value
        obs['reward'] = {
            'identity': nn.Identity(),
            'sign': torch.sign,
            'tanh': torch.tanh,
        }[self.cfg.clip_rewards](obs['reward'])
        obs['discount'] = 1.0 - obs['is_terminal'].float()
        obs['discount'] *= self.cfg.discount
        return obs

    def video_pred(self, data, key, nvid=8):
        decoder = self.heads['decoder']  # B, T, C, H, W
        truth = data[key][:nvid] + 0.5
        embed = self.encoder(data)
        states, _ = self.rssm.observe(
            embed[:nvid, :5], data['action'][:nvid, :5], data['is_first'][:nvid, :5])
        recon = decoder(self.rssm.get_feat(states))[key].mean[:nvid]  # mode
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.rssm.imagine(data['action'][:nvid, 5:], init)
        prior_recon = decoder(self.rssm.get_feat(prior))[key].mean  # mode
        model = torch.clip(torch.cat([recon[:, :5] + 0.5, prior_recon + 0.5], 1), 0, 1)
        error = (model - truth + 1) / 2

        if getattr(self, 'recon_skills', False):
            prior_feat = self.rssm.get_feat(prior)
            if self.skill_module.discrete_skills:
                B, T, _ = prior['deter'].shape
                z_e = self.skill_module.skill_encoder(prior['deter'].reshape(B * T, -1)).mean
                z_q, _ = self.skill_module.emb(z_e, weight_sg=True)
                latent_skills = z_q.reshape(B, T, -1)
            else:
                latent_skills = self.skill_module.skill_encoder(prior['deter']).mean
                latent_skills = latent_skills / torch.norm(latent_skills, dim=-1, keepdim=True)

            x = deter = self.skill_module.skill_decoder(latent_skills).mean

            stats = self.rssm._suff_stats_ensemble(x)
            index = torch.randint(0, self.rssm._ensemble, ())
            stats = {k: v[index] for k, v in stats.items()}
            dist = self.rssm.get_dist(stats)
            stoch = dist.sample()
            prior = {'stoch': stoch, 'deter': deter, **stats}
            skill_recon = decoder(self.rssm.get_feat(prior))[key].mean  # mode
            error = torch.clip(torch.cat([recon[:, :5] + 0.5, skill_recon + 0.5], 1), 0, 1)

        video = torch.cat([truth, model, error], 3)
        B, T, C, H, W = video.shape
        return video


class ContextActorCritic(common.Module):
    def __init__(self, config, act_spec, tfstep, context_dim):
        super().__init__()
        self.cfg = config
        self.act_spec = act_spec
        self.tfstep = tfstep
        self._use_amp = (config.precision == 16)
        self.device = config.device

        self.context_dim = context_dim
        inp_size = config.rssm.deter
        if config.rssm.discrete:
            inp_size += config.rssm.stoch * config.rssm.discrete
        else:
            inp_size += config.rssm.stoch

        inp_size += context_dim
        self.actor = common.MLP(inp_size, act_spec.shape[0], **self.cfg.actor)
        self.critic = common.MLP(inp_size, (1,), **self.cfg.critic)
        if self.cfg.slow_target:
            self._target_critic = common.MLP(inp_size, (1,), **self.cfg.critic)
            self._updates = 0
        else:
            self._target_critic = self.critic
        self.actor_opt = common.Optimizer('context_actor', self.actor.parameters(), **self.cfg.actor_opt,
                                          use_amp=self._use_amp)
        self.critic_opt = common.Optimizer('context_critic', self.critic.parameters(), **self.cfg.critic_opt,
                                           use_amp=self._use_amp)
        self.rewnorm = common.StreamNorm(**self.cfg.context_reward_norm, device=self.device)

    def update(self, world_model, start, is_terminal, reward_fn):
        metrics = {}
        hor = self.cfg.imag_horizon
        with common.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(enabled=self._use_amp):
                B, T, _ = start['deter'].shape
                context_pred = start['context'].reshape(B*T, -1)

                seq = world_model.imagine(self.actor, start, is_terminal, hor,
                                          task_cond=context_pred)
                seq['context'] = seq.pop('task')
                reward = reward_fn(seq)
                seq['reward'], mets1 = self.rewnorm(reward)
                mets1 = {f'skill_reward_{k}': v for k, v in mets1.items()}
                target, mets2 = self.target(seq)
                actor_loss, mets3 = self.actor_loss(seq, target)
            metrics.update(self.actor_opt(actor_loss, self.actor.parameters()))
        with common.RequiresGrad(self.critic):
            with torch.cuda.amp.autocast(enabled=self._use_amp):
                seq = {k: stop_gradient(v) for k, v in seq.items()}
                critic_loss, mets4 = self.critic_loss(seq, target)

                # start = {k: stop_gradient(v.transpose(0,1)) for k,v in start.items()}
                # start_target, _ = self.target(start)
                # critic_loss_start, _ = self.critic_loss(start, start_target)
                # critic_loss += critic_loss_start
            metrics.update(self.critic_opt(critic_loss, self.critic.parameters()))
        metrics.update(**mets1, **mets2, **mets3, **mets4)
        self.update_slow_target()  # Variables exist after first forward pass.
        return metrics

    def actor_loss(self, seq, target):  # , step):
        metrics = {}
        # Two states are lost at the end of the trajectory, one for the boostrap
        # value prediction and one because the corresponding action does not lead
        # anywhere anymore. One target is lost at the start of the trajectory
        # because the initial state comes from the replay buffer.
        policy = self.actor(stop_gradient(get_feat_ac(seq)[:-2]))
        if self.cfg.actor_grad == 'dynamics':
            objective = target[1:]
        elif self.cfg.actor_grad == 'reinforce':
            baseline = self._target_critic(get_feat_ac(seq)[:-2]).mean  # .mode()
            advantage = stop_gradient(target[1:] - baseline)
            objective = policy.log_prob(stop_gradient(seq['action'][1:-1]))[:, :, None] * advantage
        elif self.cfg.actor_grad == 'both':
            baseline = self._target_critic(get_feat_ac(seq)[:-2]).mean  # .mode()
            advantage = stop_gradient(target[1:] - baseline)
            objective = policy.log_prob(stop_gradient(seq['action'][1:-1]))[:, :, None] * advantage
            mix = utils.schedule(self.cfg.actor_grad_mix, self.tfstep)
            objective = mix * target[1:] + (1 - mix) * objective
            metrics['context_actor_grad_mix'] = mix
        else:
            raise NotImplementedError(self.cfg.actor_grad)
        ent = policy.entropy()[:, :, None]
        ent_scale = utils.schedule(self.cfg.context_actor_ent, self.tfstep)
        objective += ent_scale * ent
        weight = stop_gradient(seq['weight'])
        actor_loss = -(weight[:-2] * objective).mean()
        metrics['context_actor_ent'] = ent.mean()
        metrics['context_actor_ent_scale'] = ent_scale
        return actor_loss, metrics

    def critic_loss(self, seq, target):
        dist = self.critic(get_feat_ac(seq)[:-1])
        target = stop_gradient(target)
        weight = stop_gradient(seq['weight'])
        critic_loss = -(dist.log_prob(target)[:, :, None] * weight[:-1]).mean()
        metrics = {'context_critic': dist.mean.mean()}  # .mode().mean()}
        return critic_loss, metrics

    def target(self, seq):
        reward = seq['reward']
        disc = seq['discount']
        # print('1', seq['feat'].shape)
        # print('2', seq['context'].shape)
        value = self._target_critic(get_feat_ac(seq)).mean  # .mode()
        # Skipping last time step because it is used for bootstrapping.
        target = common.lambda_return(
            reward[:-1], value[:-1], disc[:-1],
            bootstrap=value[-1],
            lambda_=self.cfg.discount_lambda,
            axis=0)
        metrics = {}
        metrics['context_critic_slow'] = value.mean()
        metrics['context_critic_target'] = target.mean()
        return target, metrics

    def update_slow_target(self):
        if self.cfg.slow_target:
            if self._updates % self.cfg.slow_target_update == 0:
                mix = 1.0 if self._updates == 0 else float(
                    self.cfg.slow_target_fraction)
                for s, d in zip(self.critic.parameters(), self._target_critic.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1  # .assign_add(1)