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
from evaluation.insertion_deletion import InsertionDeletion
from evaluation.regression_insertion_deletion import InsertionDeletion as InsertionDeletionRegression


def parse_args():
    parser = ArgumentParser('Insertion/deletion evaluation')
    parser.add_argument('heatmap_dir', default="", help='config file of the attribution method')
    parser.add_argument('out_dir', default="", help='config file of the attribution method')
    parser.add_argument('image_path', default="", help='config file of the attribution method')
    parser.add_argument('model_path', default="", help='directory of the heatmaps')
    parser.add_argument('label_path', default="", help='directory to save the result file')
    parser.add_argument('file_name', default="insertion_deletion.json", help='directory to save the result file')
    parser.add_argument("--covid", help="covid dataset",
                        action="store_true")
    parser.add_argument("--regression", help="regression model",
                        action="store_true")
    parser.add_argument("--subset", help="evaluate over a subset",
                        action="store_true")
    args = parser.parse_args()
    return args


def evaluation(heatmap_dir, out_dir, image_path, model_path, label_path, file_name="insertion_deletion.json",
               device='cuda:0', covid=False, regression=False, subset=False):
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

    else:
        if regression:
            category_list = ["regression"]
        else:
            category_list = ['Detector01', 'Detector2', 'Detector3']

    # generate evaluation
    evaluation_metrics = ["insertion", "deletion"]
    results = {}
    insertion_auc = np.array([])
    deletion_auc = np.array([])

    for category in tqdm(category_list, desc="Categories"):
        insertion_auc_category = np.array([])
        deletion_auc_category = np.array([])

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
                POSITIVE_FINDINGS_ONLY=True,
                covid=True,
                regression=False,
                label_path=label_path,
                return_dataloader=True)

        if regression:
            evaluator = InsertionDeletionRegression(model,
                                          pixel_batch_size=25,
                                          sigma=10.)
        else:
            target = category_list.index(category)
            evaluator = InsertionDeletion(model,
                                          pixel_batch_size=25,
                                          sigma=10.)

        num_samples = 0
        np.random.seed(seed=42)
        rand_array = np.random.rand(2000) < 0.3
        for data in tqdm(dataloader, desc="Samples"):
            num_samples += 1
            if subset and not rand_array[num_samples]:
                continue
            if covid:
                if regression:
                    input, label, filename = data
                    heatmap = cv2.imread(os.path.join(heatmap_dir, filename[0]), cv2.IMREAD_GRAYSCALE)
                else:
                    input, label, filename = data
                    heatmap = cv2.imread(os.path.join(heatmap_dir, category, filename[0]), cv2.IMREAD_GRAYSCALE)
            else:
                input, label, filename, bbox = data
                heatmap = cv2.imread(os.path.join(heatmap_dir, category, filename[0]), cv2.IMREAD_GRAYSCALE)

            # skip img if the pred score is too low
            if not regression and torch.nn.functional.sigmoid(model(input.to(device))[0, target]) < 0.6:
                continue
            heatmap = torch.from_numpy(heatmap).to(device) / 255.0

            if regression:
                res_single = evaluator.evaluate(heatmap, input.squeeze().to(device))
            else:
                res_single = evaluator.evaluate(heatmap, input.squeeze().to(device), target)
            ins_auc = res_single['ins_auc']
            insertion_auc = np.append(insertion_auc, np.array(ins_auc))
            insertion_auc_category = np.append(insertion_auc_category, np.array(ins_auc))
            del_auc = res_single['del_auc']
            deletion_auc = np.append(deletion_auc, np.array(del_auc))
            deletion_auc_category = np.append(deletion_auc_category, np.array(del_auc))
        results.update({"insertion auc_{}".format(category): insertion_auc_category})
        results.update({"deletion auc_{}".format(category): deletion_auc_category})
    mean_insertion_auc = np.mean(insertion_auc)
    mean_deletion_auc = np.mean(deletion_auc)
    results.update({"insertion auc": mean_insertion_auc})
    results.update({"deletion auc": mean_deletion_auc})
    print("insertion auc: {}, deletion auc: {}".format(mean_insertion_auc, mean_deletion_auc))
    mmcv.mkdir_or_exist(out_dir)
    mmcv.dump(results, file=os.path.join(out_dir, file_name))
    return results

if __name__ == '__main__':
    args = parse_args()
    results = evaluation(args.heatmap_dir, args.out_dir, args.image_path, args.model_path, args.label_path, args.file_name,
                         covid=args.covid, regression=args.regression, subset=args.subset)
