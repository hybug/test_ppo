#!/usr/bin/python

# This script filters environmental variables and
# generate parameters to be passed to mpirun

import os
import re
from sets import Set

blacklist_wildcards = [
    #"ARNOLD_*",
    "MESOS_*",
    "METIS_*",
    "NVIDIA_*",
    "TCE_*",
    "DMLC_*"
]

blacklist_set = Set([
    # Arnold related
    "CUDA_VERSION",
    "NV_GPU",
    "SERVICE_LOAD_PATH",
    "ROOTDIR",
    "RDMA_DEVICE",
    "BOOTSTRAP",
    "CURRENT_VERSION",
    "OLDPWD",
    "PWD",
    "PS1",

    # Linux related
    "HOME",
    "HOSTNAME",
    "IS_DOCKER_ENV",
    "IS_RUNTIME",
    "LANG",
    "LANGUAGE",
    "LC_ALL",
    "LC_CTYPE",
    "LS_COLORS",
    "SHLVL",
    "TERM",
    "_"
])

whitelist_set = {
    'ARNOLD_FRAMEWORK',
    'ARNOLD_TRIAL_OWNER',
    'ARNOLD_JOB_PSM',
    'ARNOLD_JOB_ID',
    'ARNOLD_TASK_ID',
    'ARNOLD_TRIAL_ID',
    'ARNOLD_CKPT_TRIAL_ID',
    'METIS_WORKER_0_HOST'
}
# Below also appear in some of the images, but we allow
# users to overwrite them and pass them to worker proceccess
# HOROVOD_GPU_ALLREDUCE
# HOROVOD_NCCL_HOME
# HADOOP_HDFS_HOME
# JAVA_HOME
# LD_LIBRARY_PATH
# PATH

wildcards = [re.compile(x) for x in blacklist_wildcards]

if __name__ == "__main__":
    output_str = ""

    for env in os.environ:
        if env in whitelist_set:
            output_str += "-x %s " % env
            continue
        if env in blacklist_set:
            continue
        blacklisted = False
        for w in wildcards:
            if re.match(w, env):
                blacklisted = True
                break
        if not blacklisted:
            output_str += "-x %s " % env

    print(output_str)
