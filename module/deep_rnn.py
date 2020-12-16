# coding: utf-8


def deep_rnn(rnns, x, state_ins):
    ys = []
    state_outs = []

    if state_ins is None:
        state_ins = [None] * len(rnns)
    else:
        assert len(rnns) == len(state_ins)

    y = x
    for rnn, state_in in zip(rnns, state_ins):
        if state_in is None:
            y = rnn(y)
        else:
            y = rnn(y, initial_state=state_in)

        if y.__class__ == list:
            y, state_out = y[0], y[1:]
            if not len(state_out):
                state_out = None
        else:
            state_out = None

        ys.append(y)
        state_outs.append(state_out)

    return ys, state_outs
