#!/bin/bash

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch  --config_file configs/default_accelerate_config.yaml \
    train_gpt2_utility.py
