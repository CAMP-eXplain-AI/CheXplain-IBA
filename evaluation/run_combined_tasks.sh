#!/bin/bash

PATH_TO_IMAGES="/home/ubuntu/NIH_small"
PATH_TO_MODEL='/home/ubuntu/informationbottleneck/model/results/checkpoint_best'
label_path='/home/ubuntu/informationbottleneck/model/labels'
heatmap_dir='/home/ubuntu/results/filtered_mask'
out_dir='/home/ubuntu'
file_name='insertion_deletion_filtered_mask.json'

python eval_insertion_deletion.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name

file_name='sensitivity_n_filtered_mask.json'

python eval_sensitivity_n.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name