#!/bin/bash

PATH_TO_IMAGES="/home/ubuntu/NIH_small"
PATH_TO_MODEL='/home/ubuntu/informationbottleneck/model/results/checkpoint_best'
label_path='/home/ubuntu/informationbottleneck/model/labels'
heatmap_dir='/home/ubuntu/results/ib'
out_dir='~'

python eval_sensitivity_n.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path