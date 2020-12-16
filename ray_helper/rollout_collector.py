# coding: utf-8

import itertools
import os
import time
import ray
import logging
import pickle
from queue import Queue
from threading import Thread


def flatten(list_of_list):
    "Flatten one level of nesting"
    return itertools.chain.from_iterable(list_of_list)


_LAST_FREE_TIME = 0.0
_TO_FREE = []


def ray_get_and_free(object_ids):
    """
    Call ray.get and then queue the object ids for deletion.

    This function should be used whenever possible in RLlib, to optimize
    memory usage. The only exception is when an object_id is shared among
    multiple readers.

    Args:
        object_ids (ObjectID|List[ObjectID]): Object ids to fetch and free.

    Returns:
        The result of ray.get(object_ids).
    """

    free_delay_s = 10.0
    max_free_queue_size = 100

    global _LAST_FREE_TIME
    global _TO_FREE

    result = ray.get(object_ids)
    if type(object_ids) is not list:
        object_ids = [object_ids]
    _TO_FREE.extend(object_ids)

    # batch calls to free to reduce overheads
    now = time.time()
    if (len(_TO_FREE) > max_free_queue_size
            or now - _LAST_FREE_TIME > free_delay_s):
        ray.internal.free(_TO_FREE)
        _TO_FREE = []
        _LAST_FREE_TIME = now

    return result


class RolloutCollector:
    """

    just like a dataloader, helper Policy Optimizer for data fetching
    """

    def __init__(self, server_nums, ps=None, policy_evaluator_build_func=None, **kwargs):

        server_nums = int(server_nums)
        '''
        if kwargs['cpu_per_actor'] != -1:
            cpu_p_a = int(kwargs['cpu_per_actor'])
            self._inf_servers = [Evaluator.options(num_cpus=cpu_p_a).remote(kwargs) for _ in range(server_nums)]
        else:
        '''

        (model_kwargs, build_evaluator_model, env_kwargs, build_env
         ) = policy_evaluator_build_func(kwargs)
        from ray_helper.policy_evaluator import Evaluator
        kwargs['ps'] = ps
        self._inf_servers = [
            ray.remote(
                num_cpus=kwargs['cpu_per_actor'])(Evaluator).remote(
                model_func=build_evaluator_model,
                model_kwargs=model_kwargs,
                envs_func=build_env,
                env_kwargs=env_kwargs, kwargs=kwargs) for
            _ in
            range(server_nums)]
        print('inf_server start')
        self._data_gen = self._run(**kwargs)
        self.get_one_sample = lambda: self.retrieval_sample(
            next(self._run(num_returns=1, timeout=None)))

    def _run(self, **kwargs):
        '''

        :param num_returns:
        :param timeout:
        :param kwargs:
        :return:
        '''
        working_obj_ids = []
        worker_flags = [True for _ in range(len(self._inf_servers))]  # mark是否空闲
        worker_tics = [time.time() for _ in range(len(self._inf_servers))]
        objid2_idx = {}
        while True:
            for idx, flag in enumerate(worker_flags):
                if flag:
                    server = self._inf_servers[idx]
                    obj_id = server.sample.remote()
                    working_obj_ids.append(obj_id)
                    worker_flags[idx] = False
                    objid2_idx[obj_id] = idx

            ready_ids, working_obj_ids = ray.wait(working_obj_ids,
                                                  num_returns=kwargs['num_returns'],
                                                  timeout=kwargs['timeout'])

            for _id in ready_ids:
                iidx = objid2_idx[_id]
                worker_flags[iidx] = True
                ddur = time.time() - worker_tics[iidx]
                worker_tics[iidx] = time.time()
                # print('iidx:%s, dur:%s' % (iidx, ddur))
                objid2_idx.pop(_id)
            # obj_ids = [server.get_inf_res.remote() for server in self.inf_servers]
            # set flag for worker are working
            # set flag for worker done

            ## at most 22g, then will be freed mem 3.5 * 64 * 100
            yield ready_ids

    def __next__(self):
        return next(self._data_gen)

    @staticmethod
    def retrieval_sample(ready_ids):
        try:
            all_segs = ray_get_and_free(ready_ids)
        except ray.exceptions.UnreconstructableError as e:
            all_segs = []
            logging.info(str(e))
        except ray.exceptions.RayError as e:
            all_segs = []
            logging.info(str(e))

        res = []
        for idx, (obj_id, segs) in enumerate(zip(ready_ids, all_segs)):
            segs = flatten(segs)
            for idx1, seg in enumerate(segs):
                res.append(seg)
        return res


class QueueReader(Thread):
    def __init__(self,
                 sess,
                 global_queue,
                 data_collector,
                 keys,
                 dtypes,
                 shapes,
                 ):
        Thread.__init__(self)
        import tensorflow as tf
        self.daemon = True

        self.sess = sess
        self.global_queue = global_queue
        self.data_collector = data_collector

        self.keys = keys
        self.placeholders = [
            tf.placeholder(
                dtype=dtype, shape=shape
            ) for dtype, shape in zip(dtypes, shapes)]  # cant start

        self.ready_ids = Queue(maxsize=128)
        self.start_sample_flag = False
        self.retrieval_sample_trunk_size = 8
        self.enqueue_op = self.global_queue.enqueue(
            dict(zip(keys, self.placeholders)))

    def _run_sample_loop(self):
        self.sample_thread = Thread(target=self._sample_loop)
        self.sample_thread.start()

    def _sample_loop(self):
        while True:
            ready_ids = next(self.data_collector)
            for _id in ready_ids:
                self.ready_ids.put(_id)

    def enqueue(self):
        if not self.start_sample_flag:
            self._run_sample_loop()
            self.start_sample_flag = True
        ready_ids_chunk = []
        for _ in range(self.retrieval_sample_trunk_size):
            ready_ids_chunk.append(self.ready_ids.get())
        segs = self.data_collector.retrieval_sample(ready_ids_chunk)
        for seg in segs:
            fd = {self.placeholders[i]: seg[key]
                  for i, key in enumerate(self.keys)}
            self.sess.run(self.enqueue_op, fd)

    def run(self):
        while True:
            self.enqueue()


def fetch_one_structure(small_data_collector, cache_struct_path, is_head):
    """

    :param small_data_collector:
    :param cache_struct_path:
    :param is_head: for dist train, if local rank not equal zero, just wait for result
    :return:
    """
    sleep_time = 15
    if is_head:
        if os.path.exists(cache_struct_path):
            with open(cache_struct_path, 'rb') as f:
                structure = pickle.load(f)
        else:
            while True:
                segs = small_data_collector.get_one_sample()
                if len(segs) > 0:
                    seg = segs[0]
                    if seg is not None:
                        structure = seg
                        # flatten_structure = NEST.flatten(structure)
                        if structure is not None:
                            with open(cache_struct_path, 'wb') as f:
                                pickle.dump(structure, f)
                            del small_data_collector
                            break
                logging.warning("NO DATA, SLEEP %d seconds" % sleep_time)
                time.sleep(sleep_time)
    else:
        while True:
            if os.path.exists(cache_struct_path):
                with open(cache_struct_path, 'rb') as f:
                    structure = pickle.load(f)
                    if structure is not None:
                        break
            else:
                time.sleep(sleep_time)

    return structure
