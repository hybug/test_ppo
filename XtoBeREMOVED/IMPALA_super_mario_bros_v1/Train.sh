#!/bin/bash

fs_path=/mnt/mytmpfs
if [ ! -d $fs_path ];then
  mkdir $fs_path
  fi

#sudo mount -t tmpfs -o size=16g tmpfs $fs_path
#sh /opt/tiger/test_ppo/install_apt_depency.sh

python3 -u /opt/tiger/test_ppo/examples/IMPALA_super_mario_bros_v1/Trainer_PPOvtrace.py "$@"
