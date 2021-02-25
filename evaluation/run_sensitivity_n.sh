#!/bin/bash

PATH_TO_IMAGES="/home/ubuntu/NIH_small"
PATH_TO_MODEL='/home/ubuntu/informationbottleneck/model/results/checkpoint_best'
label_path='/home/ubuntu/informationbottleneck/model/labels'
heatmap_dir='/home/ubuntu/results/grad_cam'
out_dir='/home/ubuntu'
file_name='sensitivity_n.json'

python eval_sensitivity_n.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name

heatmap_dir='/home/ubuntu/results/ib_new_init'
out_dir='/home/ubuntu'
file_name='sensitivity_n_ib_new_init.json'

python eval_sensitivity_n.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name

heatmap_dir='/home/ubuntu/results/integrated_gradients'
out_dir='/home/ubuntu'
file_name='sensitivity_n_integrated_gradients.json'

python eval_sensitivity_n.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name

heatmap_dir='/home/ubuntu/results/ib'
out_dir='/home/ubuntu'
file_name='sensitivity_n_ib.json'

python eval_sensitivity_n.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name

heatmap_dir='/home/ubuntu/results/reverse_ib'
out_dir='/home/ubuntu'
file_name='sensitivity_n_reverse_ib.json'

python eval_sensitivity_n.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name

heatmap_dir='/home/ubuntu/results/filtered_mask'
out_dir='/home/ubuntu'
file_name='sensitivity_n_filtered_mask.json'

python eval_sensitivity_n.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name

######################################### regression model ###############################################
#PATH_TO_IMAGES="/home/ubuntu/BrixIAsmall"
#PATH_TO_MODEL='/home/ubuntu/informationbottleneck/model/results/regression_checkpoint_best'
#label_path='/home/ubuntu/informationbottleneck/model/labels'
#heatmap_dir='/home/ubuntu/results/mse_loss_with_target'
#out_dir='/home/ubuntu'
#file_name='sensitivity_n_mse_target.json'
#
#python eval_sensitivity_n.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name --covid --regression
#
#heatmap_dir='/home/ubuntu/results/mse_loss_maximize_score'
#out_dir='/home/ubuntu'
#file_name='sensitivity_n_maximize_score.json'
#
#python eval_sensitivity_n.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name --covid --regression
#
#
#heatmap_dir='/home/ubuntu/results/mse_loss_minimal_deviation'
#out_dir='/home/ubuntu'
#file_name='sensitivity_n_minimal_deviation.json'
#
#python eval_sensitivity_n.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name --covid --regression