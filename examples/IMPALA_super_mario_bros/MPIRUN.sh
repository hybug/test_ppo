#!/bin/bash

set -x

echo "=== Arnold MPI/Horovod entry script ==="
echo "=== contact:  zhuyibo@bytedance.com ==="

### prepare basic env
role=$ARNOLD_ROLE

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
if [[ $role == "server" ]]; then
  worker_id=$ARNOLD_ID
else
  echo 'MPIRUN  exit in'$role
  exit
fi

hostfile=/opt/tiger/hostfile_hvd
ssh_config=~/.ssh/config
rm -f $hostfile
rm -f $ssh_config
echo -e "StrictHostKeyChecking no\nUserKnownHostsFile /dev/null" >>${ssh_config}

if [ ! -z "$ARNOLD_SERVER_HOSTS" ]; then
  hosts=$ARNOLD_SERVER_HOSTS
else
  hosts=$METIS_SERVER_HOSTS
fi

IFS=','
HOST_CONFIGS=($hosts)
unset IFS
IFS=':'
WORKER0=(${HOST_CONFIGS[0]})
WORKER0_IP=${WORKER0[0]}
WORKER0_PORT=${WORKER0[1]}
worker_index=0
for config in "${HOST_CONFIGS[@]}"; do
  echo $config
  HOST_INFO=($config)
  echo "worker-${worker_index}" >>${hostfile}
  echo -e "Host worker-${worker_index}\n    Hostname ${HOST_INFO[0]}\n    Port ${HOST_INFO[1]}" >>${ssh_config}
  if [[ $worker_id == $worker_index ]]; then
    mpi_ssh_port=${HOST_INFO[1]}
  fi
  worker_index=$((worker_index+1))
done
unset IFS

### start sshd
/usr/sbin/sshd -p $mpi_ssh_port

### Take different roles action
if [[ "$ARNOLD_ID" != "0" ]]; then
  state="not started"
  while true; do
    test_worker0=`head -n 1 2>/dev/null < /dev/tcp/${WORKER0_IP}/${WORKER0_PORT}`
    if [ "$test_worker0" == "" ]; then
      if [ "$state" == "started" ]; then
        echo "worker 0 finished, exit"
        exit 0
      fi
      echo "Waiting for worker 0 to start"
    else
      state="started"
      echo "Waiting for worker 0 to finish"
    fi
    sleep 1m
  done
else
  IFS=':'
  for config in "${HOST_CONFIGS[@]}"; do
    HOST_INFO=($config)
    while true; do
      test_worker=`head -n 1 2>/dev/null < /dev/tcp/${HOST_INFO[0]}/${HOST_INFO[1]}`
      if [ "$test_worker" == "" ]; then
        echo "Waiting for worker ${HOST_INFO[0]}:${HOST_INFO[1]} to be ready"
      else
        break
      fi
      sleep 1m
    done
  done
  unset IFS
fi

process_num=$(($ARNOLD_SERVER_NUM * $ARNOLD_SERVER_GPU))
if [ $ARNOLD_SERVER_GPU -eq 8 ]; then
    mapping="--npernode $ARNOLD_SERVER_GPU \
        -bind-to socket"
else
    mapping="--npernode $ARNOLD_SERVER_GPU \
        -bind-to none"
fi

if [ "$ARNOLD_RDMA_DEVICE" != "" ]; then
    network_config="--mca pml ob1 \
        --mca btl ^openib,smcuda --mca btl_tcp_if_include eth0 \
        --mca oob_tcp_if_include eth0 \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_IB_HCA=$ARNOLD_RDMA_DEVICE:1 \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_SOCKET_IFNAME=eth0 \
        -x HOROVOD_MPI_THREADS_DISABLE=1"
else
    network_config="--mca btl ^openib,smcuda \
        --mca btl_tcp_if_exclude docker0,lo \
        -x NCCL_IB_DISABLE=1 \
        -x NCCL_SOCKET_IFNAME=eth0"
fi

mpi_env=`python /opt/tiger/test_ppo/examples/IMPALA_super_mario_bros/mpi_env.py`

### Running app, only on worker-0
mpirun --allow-run-as-root \
     -n $process_num \
     ${mapping} \
     -hostfile ${hostfile} \
     --mca orte_tmpdir_base /opt/tiger/openmpi_tmp \
     -x NCCL_DEBUG=INFO \
     ${mpi_env} \
     ${network_config} \
     $@ || exit 1

date
