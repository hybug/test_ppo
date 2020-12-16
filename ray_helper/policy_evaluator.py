import ray
import random


class Evaluator:
    """
        warp Env and Model,  so it is nearly independent with Algorithm
    """

    def __init__(self, model_func, model_kwargs, envs_func, env_kwargs, kwargs):
        import tensorflow as tf
        self.model = model_func(model_kwargs)
        self.envs = envs_func(env_kwargs)

        self.ckpt = None
        self.ckpt_dir = kwargs['ckpt_dir']
        self.load_ckpt_period = int(kwargs['load_ckpt_period'])

        self.ps = kwargs['ps']
        self.sess = tf.Session()
        self._init_policy_graph_param(self.sess)

        self._data_g = self._one_inf_step()

    def _init_policy_graph_param(self, sess):
        """

        init policy graph param, from ckpt or from ps
        :param sess:
        :return:
        """
        import tensorflow as tf
        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver = tf.train.Saver(max_to_keep=None, keep_checkpoint_every_n_hours=6)
            self.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
        print('getting init hashing from ps')
        self.ckpt = ray.get(self.ps.get_hashing.remote())

    def _one_inf_step(self):
        model = self.model
        sess = self.sess

        step_cnt = 0
        while True:
            if step_cnt % self.load_ckpt_period == 0:
                ckpt_hashing = ray.get(self.ps.get_hashing.remote())
                if self.ckpt != ckpt_hashing:
                    ws = ray.get(self.ps.pull.remote())
                    if ws is not None:
                        self.model.set_ws(self.sess, ws)
                        print('using new ckpt before:{} after: {}'.format(self.ckpt, ckpt_hashing))
                        self.ckpt = ckpt_hashing
                    else:
                        print('ws from ps is NULL')

            segs = self.envs.step(sess, model)

            if segs:
                random.shuffle(segs)
                while segs:
                    segs_return, segs = segs[:32], segs[32:]
                    yield segs_return
                step_cnt += 1

    def sample(self):
        buffer = [next(self._data_g)]
        return buffer
