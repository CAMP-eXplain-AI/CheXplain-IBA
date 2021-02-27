import argparse
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import os
from PIL import Image
from skimage.transform import resize
import numpy as np
import mmcv
from tqdm.auto import tqdm

import cxr_dataset as CXR
import merged_visualize_prediction as V

from torchray.attribution.grad_cam import grad_cam
from torchray.attribution.gradient import gradient
from torchray.attribution.excitation_backprop import excitation_backprop
from torchray.attribution.extremal_perturbation import extremal_perturbation, contrastive_reward
from captum.attr import IntegratedGradients
from IBA.pytorch import IBA

prak_dir = '/content/drive/MyDrive/Prak_MLMI'
PATH_TO_IMAGES = "/content/NIH small"
PATH_TO_COVID_IMAGES = "/content/BrixIAsmall"
PATH_TO_MODEL = prak_dir + '/model/results/checkpoint_best'
PATH_TO_COVID_MODEL = prak_dir + '/model/results/classification_checkpoint_best'
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

COVID_FINDINGS = [
    'Detector01',
    'Detector2',
    'Detector3']


def get_attribution(model, input, target, method, device, saliency_layer='features.norm5', iba_wrapper=None):
    input = input.to(device)
    input.requires_grad = True

    # get attribution
    if method == "grad_cam":
        saliency_map = grad_cam(model, input, target, saliency_layer=saliency_layer)
    elif method == "extremal_perturbation":
        saliency_map, _ = extremal_perturbation(model, input, target)
    elif method == 'ib':
        assert iba_wrapper, "Please give a iba wrapper as function parameter!"
        saliency_map = iba_wrapper.iba(model, input, target, device)
    elif method == 'reverse_ib':
        assert iba_wrapper, "Please give a iba wrapper as function parameter!"
        saliency_map = iba_wrapper.iba(model, input, target, device, reverse_lambda=True)
    elif method == "gradient":
        saliency_map = gradient(model, input, target)
    elif method == "excitation_backprop":
        saliency_map = excitation_backprop(model, input, target, saliency_layer=saliency_layer)
    elif method == "integrated_gradients":
        ig = IntegratedGradients(model)
        saliency_map, _ = ig.attribute(input, target=target, return_convergence_delta=True)
        saliency_map = saliency_map.squeeze().mean(0)

    # ib heatmap already a numpy array scaled to image size
    if method != 'ib' and method != 'reverse_ib':
        saliency_map = saliency_map.detach().cpu().numpy().squeeze()
        shape = (224, 224)
        saliency_map = resize(saliency_map, shape, order=1, preserve_range=True)
    return saliency_map


def save_attribution_map(mask, out_file=None, show=False):
    if mask.dtype in (float, np.float32, np.float16, np.float128):
        if mask.max() > 1.0:
            mask /= mask.max()
        mask = (mask * 255).astype(np.uint8)
    if show:
        plt.imshow(mask)
        plt.axis('off')

    if out_file is not None:
        dir_name = os.path.abspath(os.path.dirname(out_file))
        mmcv.mkdir_or_exist(dir_name)
        mask = Image.fromarray(mask, mode='L')
        mask.save(out_file)


def gen_attribution(dataloader, model, attribution_method, out_dir, device, covid=False, iba_wrapper=None):
    if not covid:
        category_list = FINDINGS
        for category in tqdm(category_list, desc="Categories"):

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
            for data in tqdm(dataloader, desc="Samples"):
                input, label, filename, bbox = data
                category_id = FINDINGS.index(category)
                mask = get_attribution(model, input, category_id, attribution_method, device, iba_wrapper=iba_wrapper)
                save_attribution_map(mask, out_file=os.path.join(out_dir, attribution_method, category, filename[0]))
    else:
        category_list = COVID_FINDINGS
        for category in tqdm(category_list, desc="Categories"):

            # get data inside category
            dataloader, model = V.load_data(
                PATH_TO_COVID_IMAGES,
                category,
                PATH_TO_COVID_MODEL,
                'test',
                POSITIVE_FINDINGS_ONLY=True,
                covid=True,
                label_path=LABEL_PATH,
                return_dataloader=True)

            # generate attribution map for each image inside this category
            for data in tqdm(dataloader, desc="Samples"):
                input, label, filename = data
                category_id = COVID_FINDINGS.index(category)
                mask = get_attribution(model, input, category_id, attribution_method, device, iba_wrapper=iba_wrapper)
                save_attribution_map(mask, out_file=os.path.join(out_dir, attribution_method, category, filename[0]))


class IbaWrapper():
    def __init__(self, informationbottleneck, model_loss_closure_with_target, beta=10, reverse_mask_beta=30):
        self.informationbottleneck = informationbottleneck
        self.model_loss_closure_with_target = model_loss_closure_with_target
        self.beta = beta
        self.reverse_mask_beta = reverse_mask_beta

    def iba(self, model, img, target, device, reverse_lambda=False):
        """
        Plots CXR, attribution map of CXR generated by IB
        Args:
          img: image
          target ((int)): target label
          reverse lambda: whether use reversed lambda
        """

        img = img[None].to(device)
        self.informationbottleneck.reverse_lambda = reverse_lambda
        if reverse_lambda:
            self.informationbottleneck.beta = self.reverse_mask_beta
        else:
            self.informationbottleneck.beta = self.beta

        heatmap = self.informationbottleneck.analyze(img.squeeze(0),
                                                     self.model_loss_closure_with_target(model, target, device))
        return heatmap


def softmax_crossentropy_loss_with_target(model, target, device="cuda:0"):
    def model_loss_closure(input):
        sce_loss = -torch.log_softmax(model(input), 1)[:, target].mean()
        return sce_loss

    return model_loss_closure


def binary_crossentropy_loss_with_target(model, target, device="cuda:0"):
    def model_loss_closure(input):
        loss = torch.nn.BCEWithLogitsLoss()
        bce_loss = loss(model(input), torch.tensor(target).view(1, -1).expand(10, -1).to(device).float())
        return bce_loss

    return model_loss_closure


def mse_loss_with_target(model, target, device="cuda:0"):
    def model_loss_closure(input):
        loss = torch.nn.MSELoss()
        mse_loss = loss(model(input), torch.tensor(target).to(device))
        return mse_loss

    return model_loss_closure


def mse_loss_maximize_score(model, target=None, device="cuda:0"):
    def model_loss_closure(input):
        loss = torch.nn.MSELoss()
        mse_loss = -loss(model(input), torch.tensor(0.).to(device))
        return mse_loss

    return model_loss_closure


def mse_loss_minimal_deviation(model, target=None, device="cuda:0"):
    def model_loss_closure(input):
        loss = torch.nn.MSELoss()
        mse_loss = loss(model(input), original_model(input))
        return mse_loss

    return model_loss_closure

if __name__ == "__main__":
    # TODO add argparser and add function call
    pass