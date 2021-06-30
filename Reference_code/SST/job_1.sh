#!/bin/bash
#COBALT -n 1
#COBALT -q skylake_8180
#COBALT -A Performance
#COBALT -t 6:00:00

export PATH="/home/rmaulik/anaconda3/bin:$PATH"

rm -rf Regular/*
rm -rf Var/*

source activate tf2_env

python main.py --test --var

