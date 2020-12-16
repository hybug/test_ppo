# coding: utf-8

from collections import namedtuple

from algorithm import dPPOc
from module import mse

PPOcCloss = namedtuple("PPOcCloss", ["p_loss", "v_loss"])


def dPPOcC(act, policy_logits, behavior_logits, advantage, policy_clip, vf, vf_target, value_clip, old_vf):
    a_loss = dPPOc(act=act,
                   policy_logits=policy_logits,
                   behavior_logits=behavior_logits,
                   advantage=advantage,
                   clip=policy_clip)
    c_loss = mse(y_hat=vf,
                 y_target=vf_target,
                 clip=value_clip,
                 clip_center=old_vf)
    return PPOcCloss(a_loss, c_loss)
