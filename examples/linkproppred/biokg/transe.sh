#!/bin/bash


CUDA_VISIBLE_DEVICES=0 python run.py --do_train --cuda --do_valid --do_test --evaluate_train \
  --model TransE -n 128 -b 512 -d 2000 -g 20 -a 1.0 -adv \
  -lr 0.0001 --max_steps 300000 --cpu_num 2 --test_batch_size 32

