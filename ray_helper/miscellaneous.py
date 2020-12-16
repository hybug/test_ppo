import os
import ray


def tf_model_ws(cls):
    """
    add set and load weight to tensorflow graph
    :param cls:
    :return:
    """
    import re

    class TF_Model_WS_Helper(cls):
        def __init__(self, *args, **kwargs):
            super(TF_Model_WS_Helper, self).__init__(*args, **kwargs)
            self.set_ws_ops = None
            self.ws_dphs = None

        @staticmethod
        def get_ws(sess):
            """

            :return:
            """
            import tensorflow as tf
            tvars = tf.trainable_variables()
            # sess = tf.get_default_session()
            # this line return None, dont know why

            ws = sess.run(tvars)
            names = [re.match(
                "^(.*):\\d+$", var.name).group(1) for var in tvars]
            return dict(zip(names, ws))

        def _set_ws(self, to_ws):
            """

            :param to_ws name2var
            :return: run the ops
            """
            import tensorflow as tf
            tvars = tf.trainable_variables()
            print('tvars', tvars)
            names = [re.match(
                "^(.*):\\d+$", var.name).group(1) for var in tvars]
            ops = []
            names_to_tvars = dict(zip(names, tvars))
            dphs = dict()
            for name, var in names_to_tvars.items():
                assert name in to_ws
                ph = tf.placeholder(dtype=to_ws[name].dtype, shape=to_ws[name].shape)
                dphs[name] = ph
                op = tf.assign(var, ph)
                ops.append(op)
            self.ws_dphs = dphs
            return tf.group(ops)

        def set_ws(self, sess, ws):
            if self.set_ws_ops is None:
                self.set_ws_ops = self._set_ws(ws)
            fd = dict()
            for k, v in self.ws_dphs.items():
                assert k in ws
                fd[v] = ws[k]
            sess.run(self.set_ws_ops, feed_dict=fd)

    return TF_Model_WS_Helper


def init_cluster_ray(log_to_driver=True):
    """

    connect to a exist ray cluster, if not exist init one
    :return:
    """
    server_hosts = os.getenv('ARNOLD_SERVER_HOSTS', None)
    assert server_hosts is not None
    server_ip, _ = server_hosts.split(',')[0].split(':')
    redis_port = int(os.environ['ARNOLD_RUN_ID']) % 1e4 + 6379
    ray.init(address=':'.join([server_ip, str(int(redis_port))]), log_to_driver=log_to_driver)


"""
    hdfs helper function
"""


def warp_exists(path, use_hdfs=False):
    if use_hdfs:
        import pyarrow as pa
        fs = pa.hdfs.connect()
        return fs.exists(path)
    else:
        return os.path.exists(path)


def warp_mkdir(dir_name, use_hdfs=False):
    if use_hdfs:
        import pyarrow as pa
        fs = pa.hdfs.connect()
        fs.mkdir(dir_name)
    else:
        os.mkdir(dir_name)
