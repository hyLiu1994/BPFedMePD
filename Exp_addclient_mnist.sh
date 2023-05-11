#!/bin/bash
DATASET=Mnist
ALGORITHMS=(FedPer LGFedAvg FedRep FedSOUL BPFedPD)
SIZES=(small large)
ONLY_ONE_LOCALS=(1 0)

for alg in "${ALGORITHMS[@]}"; do
  for size in "${SIZES[@]}"; do
    for only_local in "${ONLY_ONE_LOCALS[@]}"; do
      if [ "$only_local" == 1 ]; then
        ADD_NEW_CLIENTS=0
        NUM_GLOB_ITERS=10
      else
        ADD_NEW_CLIENTS=10
        NUM_GLOB_ITERS=10
      fi
      
      if [ "$alg" == "BPFedPD" ]; then
        BATCH_SIZE=100
        ZETA=1e-3
        echo "python main.py --dataset $DATASET --datasize $size --algorithm $alg --num_glob_iters $NUM_GLOB_ITERS --batch_size $BATCH_SIZE --zeta $ZETA --add_new_client $ADD_NEW_CLIENTS --only_one_local $only_local"
      else
        echo "python main.py --dataset $DATASET --datasize $size --algorithm $alg --num_glob_iters $NUM_GLOB_ITERS --add_new_client $ADD_NEW_CLIENTS --only_one_local $only_local"
      fi
    done
  done
done | xargs -I {} -P 4 bash -c "{}"
