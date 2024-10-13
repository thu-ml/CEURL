import math
from collections import OrderedDict

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from dm_env import specs

import utils
from agent.dreamer import DreamerAgent, stop_gradient
import agent.dreamer_utils as common


def get_feat_ac(seq):
    return torch.cat([seq['feat'], seq['context'], seq['skill']], dim=-1)


# feat -> (predict) context
# feat + context -> (predict) skill
class PEACDIAYN(nn.Module):
    def __init__(self, obs_dim, skill_dim, context_dim, hidden_dim):
        super().__init__()
        self.feat_net = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.ReLU())
        self.skill_pred_net = nn.Linear(hidden_dim, skill_dim)
        self.task_pred_net = nn.Linear(hidden_dim, context_dim)

        self.apply(utils.weight_init)

    def forward(self, obs):
        feat = self.feat_net(obs)
        task_pred = self.task_pred_net(feat)
        skill_pred = self.skill_pred_net(feat)
        return task_pred, skill_pred


class PEACDIAYNAgent(DreamerAgent):
    def __init__(self, update_skill_every_step, skill_dim, diayn_scale, num_init_frames,
                 task_number, task_scale, **kwargs):
        self.skill_dim = skill_dim
        self.context_dim = task_number
        self.update_skill_every_step = update_skill_every_step
        self.num_init_frames = num_init_frames
        self.diayn_scale = diayn_scale
        self.task_scale = task_scale
        super().__init__(**kwargs)
        in_dim = self.wm.inp_size
        self._task_behavior = ContextSkillActorCritic(self.cfg, self.act_spec, self.tfstep,
                                                      self.context_dim,
                                                      self.skill_dim,
                                                      discrete_skills=True).to(self.device)

        self._skill_behavior = ContextMetaCtrlAC(self.cfg, self.context_dim, self.skill_dim,
                                                 self.tfstep,
                                                 self._task_behavior,
                                                 frozen_skills=self.cfg.freeze_skills,
                                                 skill_len=int(1)).to(self.device)

        self.hidden_dim = self.cfg.diayn_hidden
        self.reward_free = True
        self.solved_meta = None

        self.peacdiayn = PEACDIAYN(in_dim, self.skill_dim, self.context_dim,
                                 self.hidden_dim).to(self.device)
        print('skill', self.skill_dim)
        print('hidden', self.hidden_dim)
        # loss criterion
        self.diayn_criterion = nn.CrossEntropyLoss()

        # optimizers
        self.peacdiayn_opt = common.Optimizer('peac_diayn', self.peacdiayn.parameters(),
                                             **self.cfg.model_opt, use_amp=self._use_amp)

        self.peacdiayn.train()
        self.requires_grad_(requires_grad=False)

    def finetune_mode(self):
        self.is_ft = True
        self.reward_free = False
        self._task_behavior.rewnorm = common.StreamNorm(**{"momentum": 1.00, "scale": 1.0, "eps": 1e-8},
                                                        device=self.device)
        self._skill_behavior.rewnorm = common.StreamNorm(**{"momentum": 1.00, "scale": 1.0, "eps": 1e-8},
                                                         device=self.device)
        self.cfg.actor_ent = 1e-4
        self.cfg.skill_actor_ent = 1e-4

    def act(self, obs, meta, step, eval_mode, state):
        obs = {k: torch.as_tensor(np.copy(v), device=self.device).unsqueeze(0) for k, v in obs.items()}
        meta = {k: torch.as_tensor(np.copy(v), device=self.device).unsqueeze(0) for k, v in meta.items()}

        if state is None:
            latent = self.wm.rssm.initial(len(obs['reward']))
            action = torch.zeros((len(obs['reward']),) + self.act_spec.shape, device=self.device)
        else:
            latent, action = state
        embed = self.wm.encoder(self.wm.preprocess(obs))
        should_sample = (not eval_mode) or (not self.cfg.eval_state_mean)
        latent, _ = self.wm.rssm.obs_step(latent, action, embed, obs['is_first'], should_sample)
        feat = self.wm.rssm.get_feat(latent)

        task_pred, skill_pred = self.peacdiayn(feat)
        context = F.softmax(task_pred, dim=-1)

        # pretrain
        if self.reward_free:
            skill = meta['skill']
            if eval_mode:
                action = self._task_behavior.actor(torch.cat([feat, context, skill], dim=-1))
                action = action.mean
            else:
                action = self._task_behavior.actor(torch.cat([feat, context, skill], dim=-1))
                action = action.sample()
            new_state = (latent, action)
            return action.cpu().numpy()[0], new_state

        # fine tune
        if eval_mode:
            skill = self._skill_behavior.actor(torch.cat([feat, context], dim=-1))
            skill = skill.mode()
            action = self._task_behavior.actor(torch.cat([feat, context, skill], dim=-1))
            action = action.mean
        else:
            skill = self._skill_behavior.actor(torch.cat([feat, context], dim=-1))
            skill = skill.sample()
            action = self._task_behavior.actor(torch.cat([feat, context, skill], dim=-1))
            action = action.sample()
        new_state = (latent, action)
        return action.cpu().numpy()[0], new_state

    def get_meta_specs(self):
        return (specs.Array((self.skill_dim,), np.float32, 'skill'),)

    def init_meta(self):
        if self.solved_meta is not None:
            return self.solved_meta
        skill = np.zeros(self.skill_dim, dtype=np.float32)
        skill[np.random.choice(self.skill_dim)] = 1.0
        meta = OrderedDict()
        meta['skill'] = skill
        return meta

    def update_meta(self, meta, global_step, time_step):
        if global_step % self.update_skill_every_step == 0:
            return self.init_meta()
        return meta

    def init_from(self, other):
        # TODO: update task model?
        # init_task = self.cfg.get('init_task', 1.0)
        init_critic = self.cfg.get('init_critic', False)
        init_actor = self.cfg.get('init_actor', True)

        # copy parameters over
        print(f"Copying the pretrained world model")
        utils.hard_update_params(other.wm.rssm, self.wm.rssm)
        utils.hard_update_params(other.wm.encoder, self.wm.encoder)
        utils.hard_update_params(other.wm.heads['decoder'], self.wm.heads['decoder'])

        utils.hard_update_params(other.peacdiayn, self.peacdiayn)

        # if init_task > 0.0:
        #     print(f"Copying the task model")
        #     utils.hard_update_params(other.wm.task_model, self.wm.task_model)

        if init_actor:
            print(f"Copying the pretrained actor")
            utils.hard_update_params(other._task_behavior.actor, self._task_behavior.actor)

        if init_critic:
            print(f"Copying the pretrained critic")
            utils.hard_update_params(other._task_behavior.critic, self._task_behavior.critic)
            if self.cfg.slow_target:
                utils.hard_update_params(other._task_behavior._target_critic, self._task_behavior._target_critic)

    def compute_diayn_loss(self, next_state, skill, embodiment_id_key=None):
        """
        DF Loss
        """
        z_hat = torch.argmax(skill, dim=1)
        task_pred, skill_pred = self.peacdiayn(next_state)
        skill_pred_log_softmax = F.log_softmax(skill_pred, dim=1)
        _, pred_z = torch.max(skill_pred_log_softmax, dim=1, keepdim=True)
        skill_loss = self.diayn_criterion(skill_pred, z_hat)
        skill_accuracy = torch.sum(
            torch.eq(z_hat,
                     pred_z.reshape(1,
                                    list(
                                        pred_z.size())[0])[0])).float() / list(
            pred_z.size())[0]

        task_loss = None
        if self.context_dim > 1:
            task_loss = F.cross_entropy(task_pred, embodiment_id_key)

        return skill_loss, skill_accuracy, task_loss

    def update_diayn(self, skill, next_obs, embodiment_id_key, step):
        B, T, _ = skill.shape
        skill = skill.reshape(B * T, -1)
        next_obs = next_obs.reshape(B * T, -1)

        metrics = dict()
        embodiment_id_key = embodiment_id_key.reshape(B*T).to(torch.int64)
        skill_loss, df_accuracy, task_loss = self.compute_diayn_loss(next_obs, skill, embodiment_id_key)

        metrics.update(self.peacdiayn_opt(skill_loss + task_loss, self.peacdiayn.parameters()))

        metrics['diayn_loss'] = skill_loss.item()
        metrics['diayn_acc'] = df_accuracy
        metrics['task_loss'] = task_loss.item()

        return metrics

    def compute_skill_reward(self, seq):
        B, T, _ = seq['skill'].shape
        skill = seq['skill'].reshape(B * T, -1)
        next_obs = seq['feat'].reshape(B * T, -1)
        z_hat = torch.argmax(skill, dim=1)

        task_pred, skill_pred = self.peacdiayn(next_obs)

        skill_pred_log_softmax = F.log_softmax(skill_pred, dim=1)
        _, pred_z = torch.max(skill_pred_log_softmax, dim=1, keepdim=True)
        skill_rew = skill_pred_log_softmax[torch.arange(skill_pred.shape[0]), z_hat] \
                    - math.log(1 / self.skill_dim)

        return skill_rew.reshape(B, T, 1) * self.diayn_scale

    def compute_context_reward(self, seq):
        B = seq['skill'].shape[0]
        T = seq['skill'].shape[1]
        next_obs = seq['feat'].reshape(B * T, -1)

        task_pred, skill_pred = self.peacdiayn(next_obs)

        task_truth = seq['embodiment_id'].repeat(B, 1, 1).to(dtype=torch.int64)

        # calculate the task model predict prob
        task_pred = F.log_softmax(task_pred, dim=1)
        intr_rew = task_pred[torch.arange(B * T), task_truth.reshape(-1)]  # B*T
        task_rew = -intr_rew.reshape(B, T, 1)

        return task_rew * self.task_scale

    def update(self, data, step):
        metrics = {}
        state, outputs, mets = self.wm.update(data, state=None)
        metrics.update(mets)
        start = outputs['post']

        start['embodiment_id'] = data['embodiment_id']
        start = {k: stop_gradient(v) for k, v in start.items()}

        feat = self.wm.rssm.get_feat(start)
        task_pred, skill_pred = self.peacdiayn(feat)
        context = F.softmax(task_pred, dim=-1).detach()
        start['context'] = stop_gradient(context)

        if self.reward_free:
            skill = data['skill']
            with common.RequiresGrad(self.peacdiayn):
                with torch.cuda.amp.autocast(enabled=self._use_amp):
                    embodiment_id_key = data['embodiment_id']
                    metrics.update(self.update_diayn(skill, feat, embodiment_id_key, step))
            reward_fn = lambda seq: self.compute_context_reward(seq) + \
                                    self.compute_skill_reward(seq)
            metrics.update(self._task_behavior.update(
                self.wm, start, data['is_terminal'], reward_fn))
        else:
            reward_fn = lambda seq: self.wm.heads['reward'](seq['feat']).mean  # .mode()
            metrics.update(self._skill_behavior.update(
                self.wm, self.peacdiayn, start, data['is_terminal'], reward_fn))
            if self.solved_meta is None:
                return state, metrics

        return state, metrics

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
            B, T, _ = seq['feat'].shape
            seq['skill'] = torch.from_numpy(self.solved_meta['skill']).repeat(B, T, 1).to(self.device)
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
        Plan next action using Dyna-MPC inference.
        """
        if self.solved_meta is None:
            return self.act(obs, meta, step, eval_mode, state)

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
            img_skill = torch.from_numpy(self.solved_meta['skill']).repeat(num_pi_trajs, 1).to(self.device)
            seq = self.wm.imagine(self._task_behavior.actor, start, None, self.cfg.mpc_opt.horizon, task_cond=img_skill)
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


# feat + context + skill -> action
class ContextSkillActorCritic(common.Module):
    def __init__(self, config, act_spec, tfstep, context_dim, skill_dim,
                 solved_meta=None, discrete_skills=True):
        super().__init__()
        self.cfg = config
        self.act_spec = act_spec
        self.tfstep = tfstep
        self._use_amp = (config.precision == 16)
        self.device = config.device

        self.discrete_skills = discrete_skills
        self.solved_meta = solved_meta

        # cat (context, skill)
        self.context_dim = context_dim
        self.skill_dim = skill_dim
        inp_size = config.rssm.deter
        if config.rssm.discrete:
            inp_size += config.rssm.stoch * config.rssm.discrete
        else:
            inp_size += config.rssm.stoch

        inp_size += context_dim
        inp_size += skill_dim
        self.actor = common.MLP(inp_size, act_spec.shape[0], **self.cfg.actor)
        self.critic = common.MLP(inp_size, (1,), **self.cfg.critic)
        if self.cfg.slow_target:
            self._target_critic = common.MLP(inp_size, (1,), **self.cfg.critic)
            self._updates = 0
        else:
            self._target_critic = self.critic
        self.actor_opt = common.Optimizer('context_skill_actor', self.actor.parameters(), **self.cfg.actor_opt,
                                          use_amp=self._use_amp)
        self.critic_opt = common.Optimizer('context_skill_critic', self.critic.parameters(), **self.cfg.critic_opt,
                                           use_amp=self._use_amp)
        self.rewnorm = common.StreamNorm(**self.cfg.context_skill_reward_norm, device=self.device)

    def update(self, world_model, start, is_terminal, reward_fn):
        metrics = {}
        hor = self.cfg.imag_horizon
        with common.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(enabled=self._use_amp):
                B, T, _ = start['deter'].shape
                context_pred = start['context'].reshape(B*T, -1)

                if self.solved_meta is not None:
                    img_skill = torch.from_numpy(self.solved_meta['skill']).repeat(B * T, 1).to(self.device)
                else:
                    if self.discrete_skills:
                        img_skill = F.one_hot(torch.randint(0, self.skill_dim,
                                                            size=(B * T,), device=self.device),
                                              num_classes=self.skill_dim).float()
                    else:
                        img_skill = torch.randn((B * T, self.skill_dim), device=self.device)
                        img_skill = img_skill / torch.norm(img_skill, dim=-1, keepdim=True)

                task_cond = torch.cat([context_pred, img_skill], dim=-1)

                seq = world_model.imagine(self.actor, start, is_terminal, hor,
                                          task_cond=task_cond)
                popped_task = seq.pop('task')
                # print('pppp', popped_task.shape)
                # print('c', self.context_dim)
                # print('s', self.skill_dim)
                # seq['context'] = popped_task[..., :self.context_dim]
                seq['context'] = popped_task[:, :, :self.context_dim]
                seq['skill'] = popped_task[:, :, self.context_dim:]
                # print('cc', seq['context'].shape)
                # print('ss', seq['skill'].shape)
                reward = reward_fn(seq)
                seq['reward'], mets1 = self.rewnorm(reward)
                mets1 = {f'context_skill_reward_{k}': v for k, v in mets1.items()}
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
            metrics['context_skill_actor_grad_mix'] = mix
        else:
            raise NotImplementedError(self.cfg.actor_grad)
        ent = policy.entropy()[:, :, None]
        ent_scale = utils.schedule(self.cfg.context_skill_actor_ent, self.tfstep)
        objective += ent_scale * ent
        weight = stop_gradient(seq['weight'])
        actor_loss = -(weight[:-2] * objective).mean()
        metrics['context_skill_actor_ent'] = ent.mean()
        metrics['context_skill_actor_ent_scale'] = ent_scale
        return actor_loss, metrics

    def critic_loss(self, seq, target):
        dist = self.critic(get_feat_ac(seq)[:-1])
        target = stop_gradient(target)
        weight = stop_gradient(seq['weight'])
        critic_loss = -(dist.log_prob(target)[:, :, None] * weight[:-1]).mean()
        metrics = {'context_skill_critic': dist.mean.mean()}  # .mode().mean()}
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
        metrics['context_skill_critic_slow'] = value.mean()
        metrics['context_skill_critic_target'] = target.mean()
        return target, metrics

    def update_slow_target(self):
        if self.cfg.slow_target:
            if self._updates % self.cfg.slow_target_update == 0:
                mix = 1.0 if self._updates == 0 else float(
                    self.cfg.slow_target_fraction)
                for s, d in zip(self.critic.parameters(), self._target_critic.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1  # .assign_add(1)


# feat + context -> (choose) skill
class ContextMetaCtrlAC(common.Module):
    def __init__(self, config, context_dim, skill_dim, tfstep, skill_executor, frozen_skills=False, skill_len=1):
        super().__init__()
        self.cfg = config

        self.context_dim = context_dim
        self.skill_dim = skill_dim
        self.tfstep = tfstep
        self.skill_executor = skill_executor
        self._use_amp = (config.precision == 16)
        self.device = config.device

        inp_size = config.rssm.deter
        if config.rssm.discrete:
            inp_size += config.rssm.stoch * config.rssm.discrete
        else:
            inp_size += config.rssm.stoch
        inp_size += self.context_dim

        actor_config = {'layers': 4, 'units': 400, 'norm': 'none', 'dist': 'trunc_normal'}
        actor_config['dist'] = 'onehot'
        self.actor = common.MLP(inp_size, skill_dim, **actor_config)
        self.critic = common.MLP(inp_size, (1,), **self.cfg.critic)
        if self.cfg.slow_target:
            self._target_critic = common.MLP(inp_size, (1,), **self.cfg.critic)
            self._updates = 0
        else:
            self._target_critic = self.critic

        self.termination = False
        self.skill_len = skill_len

        self.selector_opt = common.Optimizer('selector_actor', self.actor.parameters(), **self.cfg.actor_opt,
                                             use_amp=self._use_amp)
        self.executor_opt = common.Optimizer('executor_actor', self.skill_executor.actor.parameters(),
                                             **self.cfg.actor_opt, use_amp=self._use_amp)
        self.critic_opt = common.Optimizer('selector_critic', self.critic.parameters(), **self.cfg.critic_opt,
                                           use_amp=self._use_amp)
        self.rewnorm = common.StreamNorm(**self.cfg.reward_norm, device=self.device)

    def update(self, world_model, task_model, start, is_terminal, reward_fn):
        metrics = {}
        hor = self.cfg.imag_horizon
        with common.RequiresGrad(self.actor):
            with common.RequiresGrad(self.skill_executor.actor):
                with torch.cuda.amp.autocast(enabled=self._use_amp):
                    seq = self.selector_imagine(world_model, task_model, self.actor, start, is_terminal, hor)
                    reward = reward_fn(seq)
                    seq['reward'], mets1 = self.rewnorm(reward)
                    mets1 = {f'reward_{k}': v for k, v in mets1.items()}
                    target, mets2 = self.target(seq)
                    high_actor_loss, low_actor_loss, mets3 = self.actor_loss(seq, target)
                metrics.update(self.selector_opt(high_actor_loss, self.actor.parameters()))
                metrics.update(self.executor_opt(low_actor_loss, self.skill_executor.actor.parameters()))
        with common.RequiresGrad(self.critic):
            with torch.cuda.amp.autocast(enabled=self._use_amp):
                seq = {k: stop_gradient(v) for k, v in seq.items()}
                critic_loss, mets4 = self.critic_loss(seq, target)
            metrics.update(self.critic_opt(critic_loss, self.critic.parameters()))
        metrics.update(**mets1, **mets2, **mets3, **mets4)
        self.update_slow_target()  # Variables exist after first forward pass.
        return metrics

    def actor_loss(self, seq, target):
        self.tfstep = 0
        metrics = {}
        skill = stop_gradient(seq['skill'])
        action = stop_gradient(seq['action'])
        policy = self.actor(stop_gradient(torch.cat([seq['feat'][:-2], seq['context'][:-2]],
                                                    dim=-1)))
        low_inp = stop_gradient(torch.cat([seq['feat'][:-2], seq['context'][:-2],
                                          skill[:-2]], dim=-1))
        low_policy = self.skill_executor.actor(low_inp)
        if self.cfg.actor_grad == 'dynamics':
            low_objective = target[1:]

        ent_scale = utils.schedule(self.cfg.actor_ent, self.tfstep)
        weight = stop_gradient(seq['weight'])

        low_ent = low_policy.entropy()[:, :, None]
        high_ent = policy.entropy()[:, :, None]

        baseline = self._target_critic(torch.cat([seq['feat'][:-2], seq['context'][:-2]],
                                                 dim=-1)).mean
        advantage = stop_gradient(target[1:] - baseline)
        log_probs = policy.log_prob(skill[1:-1])[:, :, None]

        # Note: this is impactful only if skill_len > 1. In Choreographer we fixed skill_len to 1
        indices = torch.arange(0, log_probs.shape[0], step=self.skill_len, device=self.device)
        advantage = torch.index_select(advantage, 0, indices)
        log_probs = torch.index_select(log_probs, 0, indices)
        high_ent = torch.index_select(high_ent, 0, indices)
        high_weight = torch.index_select(weight[:-2], 0, indices)

        high_objective = log_probs * advantage
        if getattr(self, 'reward_smoothing', False):
            high_objective *= 0
            low_objective *= 0

        high_objective += ent_scale * high_ent
        high_actor_loss = -(high_weight * high_objective).mean()
        low_actor_loss = -(weight[:-2] * low_objective).mean()

        metrics['high_actor_ent'] = high_ent.mean()
        metrics['low_actor_ent'] = low_ent.mean()
        metrics['skills_updated'] = len(torch.unique(torch.argmax(skill, dim=-1)))
        return high_actor_loss, low_actor_loss, metrics

    def critic_loss(self, seq, target):
        dist = self.critic(torch.cat([seq['feat'][:-1], seq['context'][:-1]], dim=-1))
        target = stop_gradient(target)
        weight = stop_gradient(seq['weight'])
        critic_loss = -(dist.log_prob(target)[:, :, None] * weight[:-1]).mean()
        metrics = {'critic': dist.mean.mean()}
        return critic_loss, metrics

    def target(self, seq):
        reward = seq['reward']
        disc = seq['discount']
        value = self._target_critic(torch.cat([seq['feat'], seq['context']], dim=-1)).mean
        # Skipping last time step because it is used for bootstrapping.
        target = common.lambda_return(
            reward[:-1], value[:-1], disc[:-1],
            bootstrap=value[-1],
            lambda_=self.cfg.discount_lambda,
            axis=0)
        metrics = {}
        metrics['critic_slow'] = value.mean()
        metrics['critic_target'] = target.mean()
        return target, metrics

    def update_slow_target(self):
        if self.cfg.slow_target:
            if self._updates % self.cfg.slow_target_update == 0:
                mix = 1.0 if self._updates == 0 else float(
                    self.cfg.slow_target_fraction)
                for s, d in zip(self.critic.parameters(), self._target_critic.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1

    def selector_imagine(self, wm, task_model, policy, start, is_terminal, horizon, eval_policy=False):
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}
        start['feat'] = wm.rssm.get_feat(start)
        task_pred, skill_pred = task_model(start['feat'])
        start['context'] = F.softmax(task_pred, dim=-1)
        inp = torch.cat([start['feat'], start['context']], dim=-1)
        fake_skill = policy(inp).mean
        fake_action = self.skill_executor.actor(torch.cat([inp, fake_skill], dim=-1)).mean
        B, _ = fake_action.shape
        start['skill'] = torch.zeros_like(fake_skill, device=wm.device)
        start['action'] = torch.zeros_like(fake_action, device=wm.device)
        seq = {k: [v] for k, v in start.items()}
        for h in range(horizon):
            inp = stop_gradient(torch.cat([seq['feat'][-1], seq['context'][-1]], dim=-1))
            if h % self.skill_len == 0:
                skill = policy(inp)
                if not eval_policy:
                    skill = skill.sample()
                else:
                    skill = skill.mode()

            executor_inp = stop_gradient(torch.cat([inp, skill], dim=-1))
            action = self.skill_executor.actor(executor_inp)
            action = action.sample() if not eval_policy else action.mean
            state = wm.rssm.img_step({k: v[-1] for k, v in seq.items()}, action)
            feat = wm.rssm.get_feat(state)
            task_pred, skill_pred = task_model(feat)
            context = F.softmax(task_pred, dim=-1)
            for key, value in {**state, 'action': action, 'feat': feat, 'skill': skill,
                               'context': context, }.items():
                seq[key].append(value)
        # shape will be (T, B, *DIMS)
        seq = {k: torch.stack(v, 0) for k, v in seq.items()}
        if 'discount' in wm.heads:
            disc = wm.heads['discount'](seq['feat']).mean()
            if is_terminal is not None:
                # Override discount prediction for the first step with the true
                # discount factor from the replay buffer.
                true_first = 1.0 - flatten(is_terminal)
                true_first *= wm.cfg.discount
                disc = torch.cat([true_first[None], disc[1:]], 0)
        else:
            disc = wm.cfg.discount * torch.ones(list(seq['feat'].shape[:-1]) + [1], device=wm.device)
        seq['discount'] = disc
        # Shift discount factors because they imply whether the following state
        # will be valid, not whether the current state is valid.
        seq['weight'] = torch.cumprod(
            torch.cat([torch.ones_like(disc[:1], device=wm.device), disc[:-1]], 0), 0)
        return seq