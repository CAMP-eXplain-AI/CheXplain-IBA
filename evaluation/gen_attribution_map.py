import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import os
from PIL import Image
from skimage.transform import resize
import numpy as np
import mmcv

import cxr_dataset as CXR
import merged_visualize_prediction as V

from torchray.attribution.grad_cam import grad_cam
from torchray.attribution.guided_backprop import guided_backprop
from torchray.attribution.extremal_perturbation import extremal_perturbation, contrastive_reward
from IBA.pytorch import IBA

prak_dir = '/content/drive/MyDrive/Prak_MLMI'
PATH_TO_IMAGES = "/content/NIH small"
PATH_TO_MODEL = prak_dir + '/model/results/checkpoint_best'
LABEL_PATH = '/content/drive/MyDrive/Prak_MLMI/model/labels'

FINDINGS = [
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


def sample_dataset(dataset, num_samples=100, seed=None):
    """sample a smaller dataset from dataset"""
    np.random.seed(seed=seed)
    inds = np.random.choice(len(dataset), num_samples, replace=False)
    small_dataset = Subset(dataset, inds)
    return small_dataset


def get_attribution(model, input, target, method, device):
    input = input.to(device)
    input.requires_grad = True

    # get attribution
    if method == "grad_cam":
        saliency_map = grad_cam(model, input, target, saliency_layer='features.norm5')
    elif method == "extremal_perturbation":
        saliency_map, _ = extremal_perturbation(model, input, target)

    saliency_map = saliency_map.detach().cpu().numpy().squeeze()
    shape = (224, 224)
    saliency_map = resize(saliency_map, shape, order=1, preserve_range=True)
    return saliency_map


def save_attribution_map(mask, out_file=None):
    if mask.dtype in (float, np.float32, np.float16, np.float128):
        assert mask.max() <= 1.0
        mask = (mask * 255).astype(np.uint8)
    plt.imshow(mask)
    plt.axis('off')

    if out_file is not None:
        dir_name = os.path.abspath(os.path.dirname(out_file))
        mmcv.mkdir_or_exist(dir_name)
        mask = Image.fromarray(mask, mode='L')
        mask.save(out_file)


def gen_attribution(dataloader, model, attribution_method, out_dir, device, covid=False):
    if not covid:
        category_list = FINDINGS
        for category in category_list:

            # get data inside category
            dataloader, model = V.load_data(
                PATH_TO_IMAGES,
                category,
                PATH_TO_MODEL,
                'BBox',
                POSITIVE_FINDINGS_ONLY=True,
                label_path=LABEL_PATH,
                return_dataloader=True)

            # generate attribution map for each image inside this category
            for data in dataloader:
                input, label, filename, bbox = data
                category_id = FINDINGS.index(category)
                mask = get_attribution(model, input, category_id, attribution_method, device)
                save_attribution_map(mask, out_file=os.path.join(out_dir, attribution_method, category, filename[0]))
