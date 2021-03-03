#!/bin/bash

PATH_TO_IMAGES="/home/ubuntu/NIH_small"
PATH_TO_MODEL='/home/ubuntu/informationbottleneck/model/results/checkpoint_best_weighted'
label_path='/home/ubuntu/informationbottleneck/model/labels'

heatmap_dir='/home/ubuntu/results/nih_weighted/grad_cam'
out_dir='/home/ubuntu/nih_weighted'
file_name='sensitivity_n.json'

python eval_sensitivity_n.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name --blur

#heatmap_dir='/home/ubuntu/results/nih_weighted/ib_new_init'
#out_dir='/home/ubuntu'
#file_name='sensitivity_n_ib_new_init.json'
#
#python eval_sensitivity_n.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name --blur

heatmap_dir='/home/ubuntu/results/nih_weighted/integrated_gradients'
file_name='sensitivity_n_integrated_gradients.json'

python eval_sensitivity_n.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name --blur

heatmap_dir='/home/ubuntu/results/nih_weighted/ib'
file_name='sensitivity_n_ib.json'

python eval_sensitivity_n.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name --blur

heatmap_dir='/home/ubuntu/results/nih_weighted/reverse_ib'
file_name='sensitivity_n_reverse_ib.json'

python eval_sensitivity_n.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name --blur

#heatmap_dir='/home/ubuntu/results/nih_weighted/filtered_mask'
#out_dir='/home/ubuntu'
#file_name='sensitivity_n_filtered_mask_blur.json'
#
#python eval_sensitivity_n.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name --blur

heatmap_dir='/home/ubuntu/results/nih_weighted/gradient'
file_name='sensitivity_n_gradient.json'

python eval_sensitivity_n.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name --blur

heatmap_dir='/home/ubuntu/results/nih_weighted/excitation_backprop'
out_dir='/home/ubuntu'
file_name='sensitivity_n_excitation_backprop.json'

python eval_sensitivity_n.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name --blur


# test baseline img
#heatmap_dir='/home/ubuntu/results/constant_attribution'
#out_dir='/home/ubuntu'
#file_name='sensitivity_n_constant_blur.json'
#
#python eval_sensitivity_n.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name --blur
#
#heatmap_dir='/home/ubuntu/results/random_attribution'
#out_dir='/home/ubuntu'
#file_name='sensitivity_n_random_blur.json'
#
#python eval_sensitivity_n.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name --blur
#
#heatmap_dir='/home/ubuntu/results/constant_attribution'
#out_dir='/home/ubuntu'
#file_name='sensitivity_n_constant.json'
#
#python eval_sensitivity_n.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name
#
#heatmap_dir='/home/ubuntu/results/random_attribution'
#out_dir='/home/ubuntu'
#file_name='sensitivity_n_random.json'
#
#python eval_sensitivity_n.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name

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