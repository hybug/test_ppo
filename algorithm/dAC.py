# coding: utf-8

from collections import namedtuple

from module import IS_from_logits
from module import mse

ACloss = namedtuple("ACloss", ["p_loss", "v_loss"])


def dAC(act, policy_logits, advantage,
        vf, vf_target, value_clip=None, old_vf=None):
    ros = IS_from_logits(
        policy_logits=policy_logits,
        act=act)
    a_loss = - advantage * ros
    c_loss = mse(
        y_hat=vf,
        y_target=vf_target,
        clip=value_clip,
        clip_center=old_vf)
    return ACloss(a_loss, c_loss)
