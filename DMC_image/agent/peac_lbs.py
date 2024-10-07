import copy

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import utils
from agent.peac import PEACAgent, stop_gradient
import agent.dreamer_utils as common


class PEAC_LBSAgent(PEACAgent):
    def __init__(self, beta=1.0, **kwargs):
        super().__init__(**kwargs)

        self.reward_free = True
        self.beta = beta
        print("beta:", self.beta)

        # LBS
        # feat + context -> predict kl
        self.lbs = common.MLP(self.wm.inp_size+self.task_number, (1,),
                              **self.cfg.reward_head).to(self.device)
        self.lbs_opt = common.Optimizer('lbs', self.lbs.parameters(), **self.cfg.model_opt, use_amp=self._use_amp)
        self.lbs.train()

        self.requires_grad_(requires_grad=False)

    def update_lbs(self, outs):
        metrics = dict()
        B, T, _ = outs['feat'].shape
        feat, kl = outs['feat'].detach(), outs['kl'].detach()
        feat = feat.reshape(B * T, -1)
        kl = kl.reshape(B * T, -1)
        context = F.softmax(self.wm.task_model(feat), dim=-1).detach()

        loss = -self.lbs(torch.cat([feat, context], dim=-1)).log_prob(kl).mean()
        metrics.update(self.lbs_opt(loss, self.lbs.parameters()))
        metrics['lbs_loss'] = loss.item()
        return metrics

    def update(self, data, step):
        metrics = {}
        state, outputs, mets = self.wm.update(data, state=None)
        metrics.update(mets)
        start = outputs['post']
        start['embodiment_id'] = data['embodiment_id']
        start['context'] = outputs['context']
        start = {k: stop_gradient(v) for k, v in start.items()}

        if self.reward_free:
            with common.RequiresGrad(self.lbs):
                with torch.cuda.amp.autocast(enabled=self._use_amp):
                    metrics.update(self.update_lbs(outputs))
            reward_fn = lambda seq: self.compute_intr_reward(seq) + \
                                    self.beta * self.compute_task_reward(seq)
        else:
            reward_fn = lambda seq: self.wm.heads['reward'](seq['feat']).mean  # .mode()

        metrics.update(self._task_behavior.update(
            self.wm, start, data['is_terminal'], reward_fn))
        return state, metrics

    def compute_intr_reward(self, seq):
        context = F.softmax(self.wm.task_model(seq['feat']), dim=-1)
        return self.lbs(torch.cat([seq['feat'], context], dim=-1)).mean

    def compute_task_reward(self, seq):
        # print('we use calculated reward')
        B, T, _ = seq['feat'].shape
        task_pred = self.wm.task_model(seq['feat'])
        task_truth = seq['embodiment_id'].repeat(B, 1, 1).to(dtype=torch.int64)
        # print(task_pred.shape) # 16, 2500, task_number
        # print(seq['action'].shape) # 16, 2500, _
        # print(task_truth.shape) # 16, 2500, 1
        task_pred = F.log_softmax(task_pred, dim=2)
        task_rew = task_pred.reshape(B * T, -1)[torch.arange(B * T), task_truth.reshape(-1)]
        task_rew = -task_rew.reshape(B, T, 1)

        # print(intr_rew.shape) # 16, 2500, 1
        return task_rew
