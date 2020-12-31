#!/bin/bash

DATASET=ran1
i=0

set +x 

for rs in 1
do
	for dim in 100
	do
		for gamma in ${GAMMAS:="5"}
		do

export CUDA_VISIBLE_DEVICES=$i

nohup python run.py --do_train --cuda --do_valid --do_test --evaluate_train \
  --dataset ogbl-$DATASET --test_random_sample 500 --ntriples_eval_train 5000 \
  --model TransE -n 128 -b 512 -d $dim -g $gamma -a 1.0 -adv \
  -lr 0.0001 --max_steps 50000 --cpu_num 2 --test_batch_size 32 \
   >${0/.sh/}-$DATASET-$i.log 2>&1 &

i=$((i+1))
		done
	done
done
