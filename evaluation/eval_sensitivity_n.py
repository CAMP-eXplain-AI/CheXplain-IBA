import torch
import os
from tqdm.auto import tqdm
import numpy as np
import sys
from argparse import ArgumentParser

try:
    import IBA
except ModuleNotFoundError:
    sys.path.insert(0, '..')
    import IBA

import model.cxr_dataset as CXR
import model.merged_visualize_prediction as V

import cv2
import mmcv
from evaluation.sensitivity_n import SensitivityN


def parse_args():
    parser = ArgumentParser('Sensitivity-N evaluation')
    parser.add_argument('heatmap_dir', default="", help='config file of the attribution method')
    parser.add_argument('out_dir', default="", help='config file of the attribution method')
    parser.add_argument('image_path', default="", help='config file of the attribution method')
    parser.add_argument('model_path', default="", help='directory of the heatmaps')
    parser.add_argument('label_path', default="", help='directory to save the result file')
    parser.add_argument('file_name', default="sensitivity_n.json", help='directory to save the result file')
    args = parser.parse_args()
    return args


def evaluation(heatmap_dir, out_dir, image_path, model_path, label_path, file_name="sensitivity_n.json", device='cuda:0', covid=False):
    if not covid:
        category_list = [
            'Atelectasis',
            'Cardiomegaly',
            'Effusion',
            'Infiltration',
            'Mass',
            'Nodule',
            'Pneumonia',
            'Pneumothorax',
            'Consolidation',
            'Edema',
            'Emphysema',
            'Fibrosis',
            'Pleural_Thickening',
            'Hernia']

        # generate evaluation
        log_list = np.logspace(0, 4.7, num=50)
        results = {}

        for n in tqdm(log_list):
            score_diffs_all = []
            sum_attrs_all = []
            for category in category_list:

                # get data inside category
                dataloader, model = V.load_data(
                    image_path,
                    category,
                    model_path,
                    'BBox',
                    POSITIVE_FINDINGS_ONLY=True,
                    label_path=label_path,
                    return_dataloader=True)

                target = category_list.index(category)
                evaluator = SensitivityN(model, (224, 224), int(n))

                for data in dataloader:
                    input, label, filename, bbox = data

                    heatmap = cv2.imread(os.path.join(heatmap_dir, category, filename[0]), cv2.IMREAD_UNCHANGED)
                    heatmap = torch.from_numpy(heatmap).to(device) / 255.0

                    res_single = evaluator.evaluate(heatmap, input.squeeze().to(device), target)
                    score_diffs = res_single['score_diffs']
                    sum_attrs = res_single['sum_attributions']
                    score_diffs_all.append(score_diffs)
                    sum_attrs_all.append(sum_attrs)
            score_diffs_all = np.concatenate(score_diffs_all, 0)
            sum_attrs_all = np.concatenate(sum_attrs_all, 0)
            corr_matrix = np.corrcoef(score_diffs_all, sum_attrs_all)
            results.update({n: corr_matrix[1, 0]})
            print("corr for {} is {}".format(n, corr_matrix[1, 0]))
        mmcv.dump(results, file=os.path.join(out_dir, file_name))
    return results

if __name__ == '__main__':
    args = parse_args()
    results = evaluation(args.heatmap_dir, args.out_dir, args.image_path, args.model_path, args.label_path, args.file_name)