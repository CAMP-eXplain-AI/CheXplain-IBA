######################################## regression model ###############################################
PATH_TO_IMAGES="/home/ubuntu/BrixIAsmall"
PATH_TO_MODEL='/home/ubuntu/informationbottleneck/model/results/classification_checkpoint_best_weighted'
label_path='/home/ubuntu/informationbottleneck/model/labels'

heatmap_dir='/home/ubuntu/results/covid_classification_weighted/grad_cam'
out_dir='/home/ubuntu/covid_classification_weighted'
file_name='insertion_deletion.json'

python eval_insertion_deletion.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name --covid

heatmap_dir='/home/ubuntu/results/covid_classification_weighted/excitation_backprop'
file_name='insertion_deletion_excitation_backprop.json'

python eval_insertion_deletion.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name --covid

heatmap_dir='/home/ubuntu/results/covid_classification_weighted/gradient'
file_name='insertion_deletion_gradient.json'

python eval_insertion_deletion.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name --covid

heatmap_dir='/home/ubuntu/results/covid_classification_weighted/integrated_gradients'
out_dir='/home/ubuntu/covid_classification_weighted'
file_name='insertion_deletion_integrated_gradients.json'

python eval_insertion_deletion.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name --covid

heatmap_dir='/home/ubuntu/results/covid_classification_weighted/ib'
file_name='insertion_deletion_ib.json'

python eval_insertion_deletion.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name --covid

heatmap_dir='/home/ubuntu/results/covid_classification_weighted/reverse_ib'
file_name='insertion_deletion_reverse_ib.json'

python eval_insertion_deletion.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name --covid


################################################## Sensitivity N #################################################################
heatmap_dir='/home/ubuntu/results/covid_classification_weighted/grad_cam'
file_name='sensitivity_n.json'

python eval_sensitivity_n.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name --covid

heatmap_dir='/home/ubuntu/results/covid_classification_weighted/excitation_backprop'
file_name='sensitivity_n_excitation_backprop.json'

python eval_sensitivity_n.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name --covid

heatmap_dir='/home/ubuntu/results/covid_classification_weighted/gradient'
file_name='sensitivity_n_gradient.json'

python eval_sensitivity_n.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name --covid

heatmap_dir='/home/ubuntu/results/covid_classification_weighted/integrated_gradients'
file_name='sensitivity_n_integrated_gradients.json'

python eval_sensitivity_n.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name --covid

heatmap_dir='/home/ubuntu/results/covid_classification_weighted/ib'
file_name='sensitivity_n_ib.json'

python eval_sensitivity_n.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name --covid

heatmap_dir='/home/ubuntu/results/covid_classification_weighted/reverse_ib'
file_name='sensitivity_n_reverse_ib.json'

python eval_sensitivity_n.py $heatmap_dir $out_dir $PATH_TO_IMAGES $PATH_TO_MODEL $label_path $file_name --covid