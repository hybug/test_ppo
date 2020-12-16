#!/bin/bash

SCRIPT_DIR='/opt/tiger/test_ppo/examples/PPO_super_mario_bros'
PROJECT_DIR="/opt/tiger/test_ppo"
PlASMA_DIR="/dev/shm"

GAMEBOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

set -x

### prepare basic env
role=$ARNOLD_ROLE
worker_num=$ARNOLD_WORKER_NUM

### prepare env for mpi
chown root:root /root
rm -rf /var/run/sshd
rm -rf ~/.ssh/
mkdir -p /var/run/sshd
mkdir -p ~/.ssh/
cp /opt/tiger/arnold/arnold_entrypoint/tools/id_rsa ~/.ssh/
cp /opt/tiger/arnold/arnold_entrypoint/tools/id_rsa.pub ~/.ssh/authorized_keys
chmod 600 ~/.ssh/id_rsa

worker_id=-1
if [[ $role == "worker" ]]; then
  worker_id=$ARNOLD_ID
else
  server_hosts=${ARNOLD_SERVER_HOSTS}
  IFS=':' SERVER_CONFIGS=($server_hosts)
  unset IFS
  mpi_ssh_port=${SERVER_CONFIGS[1]}
fi

server_hosts=${ARNOLD_SERVER_HOSTS}
if [[ $server_hosts =~ "," ]]
then
echo 'may be in using vscode'
IFS=',' server_hosts=($server_hosts)
unset IFS
fi
IFS=':' SERVER_CONFIGS=($server_hosts)
unset IFS
SERVER_IP=${SERVER_CONFIGS[0]}
SERVER_PORT=${SERVER_CONFIGS[1]}
echo $SERVER_IP


hostfile=/opt/tiger/hostfile
ssh_config=~/.ssh/config
rm -f $hostfile
rm -f $ssh_config
echo -e "StrictHostKeyChecking no\nUserKnownHostsFile /dev/null" >>${ssh_config}

if [ ! -z "$ARNOLD_WORKER_HOSTS" ]; then
  hosts=$ARNOLD_WORKER_HOSTS
else
  hosts=$METIS_WORKER_HOSTS
fi

IFS=','
HOST_CONFIGS=($hosts)
unset IFS
IFS=':'
worker_index=0
for config in "${HOST_CONFIGS[@]}"; do
  echo $config
  HOST_INFO=($config)
  echo "worker-${worker_index}" >>${hostfile}
  echo -e "Host worker-${worker_index}\n    Hostname ${HOST_INFO[0]}\n    Port ${HOST_INFO[1]}" >>${ssh_config}
  if [[ $worker_id == $worker_index ]]; then
    mpi_ssh_port=${HOST_INFO[1]}
  fi
  worker_index=$((worker_index + 1))
done
unset IFS

REDIS_PORT=$((${ARNOLD_RUN_ID} % 10000 + 6379))
HEAD_IP=$SERVER_IP
HEAD_PORT=$SERVER_PORT
export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR
# /etc/profile
if [[ $role != "server" ]]; then
  while true; do
    test_head=$(head -n 1 2>/dev/null </dev/tcp/${HEAD_IP}/${HEAD_PORT})
    if [ "$test_head" == "" ]; then
      echo "Waiting for head ${HEAD_IP}:${HEAD_PORT} to be ready"
    else
      echo "The head is now up, start to launch ray on this node."
      break
    fi
    sleep 10s
  done
  # wait for head to actually be ready
  sleep $((20 * $worker_num / 8 + $mpi_ssh_port % 20 * 2))s
  while true; do
    if ray start --address=${SERVER_IP}:${REDIS_PORT} --num-cpus=${ARNOLD_WORKER_CPU} --memory=$(($ARNOLD_WORKER_MEM * 1024 ** 2)) --object-store-memory=$((8 * 1024 ** 3)) 2>/dev/null; then
      echo "Successfully joined the cluster."
      ### start sshd
      /usr/sbin/sshd -p $mpi_ssh_port
      break
    else
      echo "It seems that Head is not up yet. Waiting for Head to start the cluster"
    fi
    sleep 3s
  done
  echo "Successfully joined the cluster."
  ### sleep forever
  while true; do
    sleep 86400
  done
else
  cpu_remain_for_head=4
  ray start --head --include-webui --node-ip-address=$SERVER_IP --redis-port=${REDIS_PORT} --num-cpus=$((ARNOLD_SERVER_CPU - cpu_remain_for_head)) --memory=$(($ARNOLD_SERVER_MEM * 1024 ** 2))  --plasma-directory $PlASMA_DIR
  echo "The Head is running"
  ### start sshd
  /usr/sbin/sshd -p $mpi_ssh_port
  IFS=':'
  for config in "${HOST_CONFIGS[@]}"; do
    HOST_INFO=($config)
    while true; do
      test_worker=$(head -n 1 2>/dev/null </dev/tcp/${HOST_INFO[0]}/${HOST_INFO[1]})
      if [ "$test_worker" == "" ]; then
        echo "Waiting for worker ${HOST_INFO[0]}:${HOST_INFO[1]} to be ready"
      else
        break
      fi
      sleep 3s
    done
  done
  unset IFS
  echo "All worker is up. Start to run the task."
  sleep 10s
  nof_evaluator_on_worker=$((ARNOLD_WORKER_NUM * ARNOLD_WORKER_CPU / 6))
  nof_evaluator_on_server=$(((ARNOLD_SERVER_CPU - 4 - 1) / 6))

  if ((nof_evaluator_on_server <= 0))
  then
    nof_evaluator_on_server=0
  fi
  nof_evaluator=$((nof_evaluator_on_server + nof_evaluator_on_worker))

  python3 $SCRIPT_DIR/ray_trainer.py "$@" --nof_evaluator $nof_evaluator
  echo 'python3 '$SCRIPT_DIR'/ray_trainer.py '"$@" > /opt/tiger/test_ppo/cache_run_script.sh

  while true; do
    sleep 86400
  done
fi
