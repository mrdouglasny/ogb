#!/bin/bash

DIR=log/ogbl-wikikg/PairRE/400-8.0/1605147115.2863047
OUT=data/all
mkdir -p $OUT

CUDA_VISIBLE_DEVICES=5 python run.py --cuda --do_test --test_random_sample 40000 \
            --model PairRE -n 128 -b 512 -d 400 -g 8 -a 1.0 -adv -dr \
  -lr 0.0001 --max_steps 0 --cpu_num 2 --test_batch_size 32 \
  --init $DIR --evaluator mrr2 >$OUT/pair400_all.txt
