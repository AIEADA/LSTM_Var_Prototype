#!/bin/bash +x

# User Configuration
EXP_DIR=$PWD
CPUS_PER_NODE=8
GPUS_PER_NODE=8

# Initialization of environment
source /lus/theta-fs0/software/thetagpu/conda/2021-06-28/mconda3/setup.sh
conda activate /lus/eagle/projects/datascience/rmaulik/LSTM_Var_Prototype/AIAEDA

# Collect IP addresses of available compute nodes
mapfile -t nodes_array -d '\n' < $COBALT_NODEFILE
HEAD_NODE=${nodes_array[0]}
HEAD_NODE_IP=$(dig $HEAD_NODE a +short | awk 'FNR==2')
echo "Detected HEAD node $HEAD_NODE with IP $HEAD_NODE_IP"

WORKER_NODES=${nodes_array[@]:1}
for ((i=0; i < ${#WORKER_NODES[@]}; i++)); do
    WORKER_NODES_IPS[$i]=$(dig ${WORKER_NODES[$i]} a +short | awk 'FNR==1')
done
echo "Detected ${#WORKER_NODES[@]} workers with IPs: ${WORKER_NODES_IPS[@]}"

# Launch the Ray cluster
# Starting the Ray Head Node
RAY_PORT=6379

echo "Starting HEAD at $HEAD_NODE_IP"
ssh -tt $HEAD_NODE_IP "source /lus/theta-fs0/software/thetagpu/conda/2021-06-28/mconda3/setup.sh && \
                   conda activate /lus/eagle/projects/datascience/rmaulik/LSTM_Var_Prototype/AIAEDA; \
    ray start --head --node-ip-address=$HEAD_NODE_IP --port=$RAY_PORT \
    --num-cpus $CPUS_PER_NODE --num-gpus $GPUS_PER_NODE" &

# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
for ((i=0; i < ${#WORKER_NODES_IPS[@]}; i++)); do
    echo "Starting WORKER $i at ${WORKER_NODES_IPS[$i]}"
    ssh -tt ${WORKER_NODES_IPS[$i]} "source $INIT_SCRIPT && \
        ray start --address $HEAD_NODE_IP:$RAY_PORT \
        --num-cpus $CPUS_PER_NODE --num-gpus $GPUS_PER_NODE" &
    sleep 5
done

ssh $HEAD_NODE_IP "cd $EXP_DIR && source /lus/theta-fs0/software/thetagpu/conda/2021-06-28/mconda3/setup.sh && \
                   conda activate /lus/eagle/projects/datascience/rmaulik/LSTM_Var_Prototype/AIAEDA && \
                   python source/parallel_eval.py"


# Stop Ray cluster
for ((i=0; i < ${#WORKER_NODES_IPS[@]}; i++)); do
    echo "Stopping WORKER $i at ${WORKER_NODES_IPS[$i]}"
    ssh -tt ${WORKER_NODES_IPS[$i]} "source $INIT_SCRIPT && ray stop"
    sleep 5
done
