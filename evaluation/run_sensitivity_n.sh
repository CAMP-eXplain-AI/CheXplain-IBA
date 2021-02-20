#!/bin/bash

PATH_TO_IMAGES="~/NIH_small"
PATH_TO_MODEL='~/informationbottleneck/model/results/checkpoint_best'
label_path='~/informationbottleneck/model/labels'
heatmap_dir='~/results/ib'
out_dir='~'

python eval_sensitivity_n.py