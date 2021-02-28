######################################## regression model ###############################################
PATH_TO_IMAGES="/home/ubuntu/BrixIAsmall"
PATH_TO_MODEL='/home/ubuntu/informationbottleneck/model/results/regression_checkpoint_best_weighted'
label_path='/home/ubuntu/informationbottleneck/model/labels'
heatmap_dir='/home/ubuntu/results/regression_weighted/mse_loss_with_target'
out_dir='/home/ubuntu/regression'
file_name='insertion_deletion_mse_target.json'

python eval_insertion_deletion.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name --covid --regression

heatmap_dir='/home/ubuntu/results/regression_weighted/mse_loss_maximize_score'
file_name='insertion_deletion_maximize_score.json'

python eval_insertion_deletion.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name --covid --regression

heatmap_dir='/home/ubuntu/results/regression_weighted/mse_loss_minimal_deviation'
file_name='insertion_deletion_minimal_deviation.json'

python eval_insertion_deletion.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name --covid --regression

heatmap_dir='/home/ubuntu/results/regression_weighted/mse_loss_with_target'
file_name='sensitivity_n_mse_target.json'

python eval_sensitivity_n.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name --covid --regression

heatmap_dir='/home/ubuntu/results/regression_weighted/mse_loss_maximize_score'
file_name='sensitivity_n_maximize_score.json'

python eval_sensitivity_n.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name --covid --regression

heatmap_dir='/home/ubuntu/results/regression_weighted/mse_loss_minimal_deviation'
file_name='sensitivity_n_minimal_deviation.json'

python eval_sensitivity_n.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name --covid --regression