#!/bin/bash

PATH_TO_IMAGES="/home/ubuntu/NIH_small"
PATH_TO_MODEL='/home/ubuntu/informationbottleneck/model/results/checkpoint_best'
label_path='/home/ubuntu/informationbottleneck/model/labels'
heatmap_dir='/home/ubuntu/results/grad_cam'
out_dir='/home/ubuntu'
file_name='insertion_deletion.json'

python eval_insertion_deletion.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name

heatmap_dir='/home/ubuntu/results/ib'
out_dir='/home/ubuntu'
file_name='insertion_deletion_ib.json'

python eval_insertion_deletion.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name


heatmap_dir='/home/ubuntu/results/reverse_ib'
out_dir='/home/ubuntu'
file_name='insertion_deletion_reverse_ib.json'

python eval_insertion_deletion.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name
