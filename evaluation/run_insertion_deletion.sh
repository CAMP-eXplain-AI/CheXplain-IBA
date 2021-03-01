#!/bin/bash

PATH_TO_IMAGES="/home/ubuntu/NIH_small"
PATH_TO_MODEL='/home/ubuntu/informationbottleneck/model/results/checkpoint_best'
label_path='/home/ubuntu/informationbottleneck/model/labels'
#heatmap_dir='/home/ubuntu/results/nih_weighted/grad_cam'
#out_dir='/home/ubuntu'
#file_name='insertion_deletion.json'
#
#python eval_insertion_deletion.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name
#
##heatmap_dir='/home/ubuntu/results/ib_new_init'
##out_dir='/home/ubuntu'
##file_name='insertion_deletion_ib_new_init.json'
##
##python eval_insertion_deletion.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name
#
#
#heatmap_dir='/home/ubuntu/results/nih_weighted/ib'
#out_dir='/home/ubuntu'
#file_name='insertion_deletion_ib.json'
#
#python eval_insertion_deletion.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name
#
#heatmap_dir='/home/ubuntu/results/nih_weighted/reverse_ib'
#out_dir='/home/ubuntu'
#file_name='insertion_deletion_reverse_ib.json'
#
#python eval_insertion_deletion.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name
#
#heatmap_dir='/home/ubuntu/results/nih_weighted/integrated_gradients'
#out_dir='/home/ubuntu'
#file_name='insertion_deletion_integrated_gradients.json'
#
#python eval_insertion_deletion.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name
#
##heatmap_dir='/home/ubuntu/results/filtered_mask'
##out_dir='/home/ubuntu'
##file_name='insertion_deletion_filtered_mask.json'
##
##python eval_insertion_deletion.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name
#
#heatmap_dir='/home/ubuntu/results/nih_weighted/gradient'
#out_dir='/home/ubuntu'
#file_name='insertion_deletion_gradient.json'
#
#python eval_insertion_deletion.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name
#
#heatmap_dir='/home/ubuntu/results/nih_weighted/excitation_backprop'
#out_dir='/home/ubuntu'
#file_name='insertion_deletion_excitation_backprop.json'
#
#python eval_insertion_deletion.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name
######################################### regression model ###############################################
#PATH_TO_IMAGES="/home/ubuntu/BrixIAsmall"
#PATH_TO_MODEL='/home/ubuntu/informationbottleneck/model/results/regression_checkpoint_best'
#label_path='/home/ubuntu/informationbottleneck/model/labels'
#heatmap_dir='/home/ubuntu/results/mse_loss_with_target'
#out_dir='/home/ubuntu'
#file_name='insertion_deletion_mse_target.json'
#
#python eval_insertion_deletion.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name --covid --regression
#
#heatmap_dir='/home/ubuntu/results/mse_loss_maximize_score'
#out_dir='/home/ubuntu'
#file_name='insertion_deletion_maximize_score.json'
#
#python eval_insertion_deletion.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name --covid --regression
#
#
#heatmap_dir='/home/ubuntu/results/mse_loss_minimal_deviation'
#out_dir='/home/ubuntu'
#file_name='insertion_deletion_minimal_deviation.json'
#
#python eval_insertion_deletion.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name --covid --regression

######################################### regression model ###############################################
heatmap_dir='/home/ubuntu/results/filtered_mask_new_weighted_0.0020'
out_dir='/home/ubuntu'
file_name='insertion_deletion_filtered_mask02_weighted.json'

python eval_insertion_deletion.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name

heatmap_dir='/home/ubuntu/results/filtered_mask_new_weighted_0.0030'
out_dir='/home/ubuntu'
file_name='insertion_deletion_filtered_mask03_weighted.json'

python eval_insertion_deletion.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name

heatmap_dir='/home/ubuntu/results/filtered_mask_new_weighted_0.0040'
out_dir='/home/ubuntu'
file_name='insertion_deletion_filtered_mask04_weighted.json'

python eval_insertion_deletion.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name

heatmap_dir='/home/ubuntu/results/filtered_mask_new_weighted_0.0060'
out_dir='/home/ubuntu'
file_name='insertion_deletion_filtered_mask06_weighted.json'

python eval_insertion_deletion.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name