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
from evaluation.regression_sensitivity_n import SensitivityN as SensitivityNRegression

def parse_args():
    parser = ArgumentParser('Sensitivity-N evaluation')
    parser.add_argument('heatmap_dir', default="", help='config file of the attribution method')
    parser.add_argument('out_dir', default="", help='config file of the attribution method')
    parser.add_argument('image_path', default="", help='config file of the attribution method')
    parser.add_argument('model_path', default="", help='directory of the heatmaps')
    parser.add_argument('label_path', default="", help='directory to save the result file')
    parser.add_argument('file_name', default="sensitivity_n.json", help='directory to save the result file')
    parser.add_argument("--covid", help="covid dataset",
                        action="store_true")
    parser.add_argument("--regression", help="regression model",
                        action="store_true")
    parser.add_argument("--blur", help="use blurred image as baseline",
                        action="store_true")
    parser.add_argument("--sigma", default=4., help="sigma for gaussian blur")
    args = parser.parse_args()
    return args


def evaluation(heatmap_dir, out_dir, image_path, model_path, label_path, file_name="sensitivity_n.json",
               device='cuda:0', covid=False, regression=False, blur=False, sigma=4.):
    if not covid:
        category_list = [
            'Atelectasis',
            'Cardiomegaly',
            'Effusion',
            'Infiltration',
            'Mass',
            'Nodule',
            # 'Pneumonia',
            'Pneumothorax',
            'Consolidation',
            'Edema',
            'Emphysema',
            'Fibrosis',
            'Pleural_Thickening',
            'Hernia']
    else:
        category_list = ["regression"]

    # generate evaluation
    log_list = np.logspace(0, 4.7, num=50)
    results = {}

    passed_n = 0
    for n in tqdm(log_list):
        passed_n += 1
        score_diffs_all = []
        sum_attrs_all = []
        corr_all = np.array([])
        for category in category_list:
            corr_category = np.array([])

            # get data inside category
            if not covid:
                dataloader, model = V.load_data(
                    image_path,
                    category,
                    model_path,
                    'BBox',
                    POSITIVE_FINDINGS_ONLY=True,
                    label_path=label_path,
                    return_dataloader=True)

            elif covid and regression:
                dataloader, model = V.load_data(
                    image_path,
                    category,
                    model_path,
                    'test',
                    POSITIVE_FINDINGS_ONLY=False,
                    covid=True,
                    regression=True,
                    label_path=label_path,
                    return_dataloader=True)

            elif covid and not regression:
                dataloader, model = V.load_data(
                    image_path,
                    category,
                    model_path,
                    'test',
                    POSITIVE_FINDINGS_ONLY=False,
                    covid=True,
                    regression=False,
                    label_path=label_path,
                    return_dataloader=True)

            if regression:
                evaluator = SensitivityNRegression(model, (224, 224), int(n))
            else:
                target = category_list.index(category)
                evaluator = SensitivityN(model, (224, 224), int(n))

            for data in dataloader:
                if covid:
                    input, label, filename = data
                    heatmap = cv2.imread(os.path.join(heatmap_dir, filename[0]), cv2.IMREAD_GRAYSCALE)
                else:
                    input, label, filename, bbox = data
                    heatmap = cv2.imread(os.path.join(heatmap_dir, category, filename[0]), cv2.IMREAD_GRAYSCALE)
                heatmap = torch.from_numpy(heatmap).to(device) / 255.0

                if regression:
                    res_single = evaluator.evaluate(heatmap, input.squeeze().to(device), calculate_corr=True)
                else:
                    res_single = evaluator.evaluate(heatmap, input.squeeze().to(device), target, calculate_corr=True)
                corr = res_single['correlation']

                # manually set NaN to zero
                if np.isnan(corr):
                    corr = 0.
                # score_diffs = res_single['score_diffs']
                # sum_attrs = res_single['sum_attributions']
                # score_diffs_all.append(score_diffs)
                # sum_attrs_all.append(sum_attrs)
                corr_category = np.append(corr_category, np.array([corr]))
                corr_all = np.append(corr_all, np.array([corr]))
            results.update({"{}_{}".format(passed_n, category): corr_category})
        # score_diffs_all = np.concatenate(score_diffs_all, 0)
        # sum_attrs_all = np.concatenate(sum_attrs_all, 0)
        # corr_matrix = np.corrcoef(score_diffs_all, sum_attrs_all)
        # results.update({n: corr_matrix[1, 0]})
        corr_mean = corr_all.mean()
        results.update({n: corr_mean})
        print("corr for {} is {}".format(n, corr_mean))
    mmcv.mkdir_or_exist(out_dir)
    mmcv.dump(results, file=os.path.join(out_dir, file_name))
    return results

if __name__ == '__main__':
    args = parse_args()
    results = evaluation(args.heatmap_dir, args.out_dir, args.image_path, args.model_path, args.label_path, args.file_name,
                         covid=args.covid, regression=args.regression, blur=args.blur, sigma=args.sigma)