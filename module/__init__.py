# coding: utf-8

from module.mse import mse
from module.icm import icm
from module.coex import coex
from module.vtrace import from_logits as vtrace_from_logits
from module.retrace import from_logits as retrace_from_logits
from module.gae import gae
from module.IS import IS, IS_from_logits
from module.KLDiv import KL_from_logits, KL_from_gaussians
from module.entropy import entropy_from_logits
from module.doubleQ import doubleQ
from module.duelingQ import duelingQ
from module.rescaleTarget import rescaleTarget
from module.dropout import dropout
from module.gelu import gelu
from module.attention import attention
from module.RMCRNN import RMCRNN, RMCRNNCell
from module.TmpHierRMCRNN import TmpHierRMCRNN, TmpHierRMCRNNCell
from module.TmpHierRMCRNN_v2 import TmpHierRMCRNN_v2, TmpHierRMCRNNCell_v2
from module.TmpHierRNN import TmpHierRNN, TmpHierRNNCell
from module.AMCRNN import AMCRNN, AMCRNNCell
from module.deep_rnn import deep_rnn
from module.NewLSTM import NewLSTM, NewLSTMCell

version = "1.14.1"
