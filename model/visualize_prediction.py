from __future__ import print_function, division

# pytorch imports
import torch
from torchray.attribution.grad_cam import grad_cam
from torchvision import transforms

# image / graphics imports
from PIL import Image
from pylab import *
import seaborn as sns
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# data science
import numpy as np
import pandas as pd
from scipy import ndimage

# import other modules
import cxr_dataset as CXR

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def calc_cam(x, label, model):
    """
    function to generate a class activation map corresponding to a torch image tensor

    Args:
        x: the 1x3x224x224 pytorch tensor file that represents the NIH CXR
        label:user-supplied label you wish to get class activation map for; must be in FINDINGS list
        model: densenet121 trained on NIH CXR data

    Returns:
        cam_torch: 224x224 torch tensor containing activation map
    """
    FINDINGS = [
        'Atelectasis',
        'Cardiomegaly',
        'Effusion',
        'Infiltrate',
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

    if label not in FINDINGS:
        raise ValueError(
            str(label) +
            "is an invalid finding - please use one of " +
            str(FINDINGS))

    # find index for label; this corresponds to index from output of net
    label_index = next(
        (x for x in range(len(FINDINGS)) if FINDINGS[x] == label))

    # define densenet_last_layer class so we can get last 1024 x 7 x 7 output
    # of densenet for class activation map
    class densenet_last_layer(torch.nn.Module):
        def __init__(self, model):
            super(densenet_last_layer, self).__init__()

            DenseNet_Module = list(model.children())[0]
            self.features = torch.nn.Sequential(
                *list(DenseNet_Module.children())[:-1]
            )

        def forward(self, x):
            x = self.features(x)
            x = torch.nn.functional.relu(x)
            return x

    # instantiate cam model and get output
    model_cam = densenet_last_layer(model)
    y = model_cam(x)
    y = y.cpu().data.numpy()
    y = np.squeeze(y)

    # pull weights corresponding to the 1024 layers from model
    weights = model.state_dict()['classifier.weight']
    weights = weights.cpu().numpy()
    
    bias = model.state_dict()['classifier.bias']
    bias = bias.cpu().numpy()

    # Calculating cams for all classes at the same time, however for the bounding box test data set
    # it would be of no use, since each image only has one class
    # weights = weights[:, :, np.newaxis, np.newaxis]
    # cams = np.multiply(weights, y)
    # cams = np.sum(cams, axis=1).squeeze()
    # bias = bias[:, np.newaxis, np.newaxis]
    # cams += bias
    # cam = cams[label_index]

    class_weights = weights[label_index]
    class_weights = class_weights[:, np.newaxis, np.newaxis]
    cam = np.multiply(class_weights, y)
    cam = np.sum(cam, axis=0).squeeze()
    cam += bias[label_index]

    return cam


def load_data(
        PATH_TO_IMAGES,
        LABEL,
        PATH_TO_MODEL,
        fold,
        POSITIVE_FINDINGS_ONLY=None,
        covid=False):
    """
    Loads dataloader and torchvision model

    Args:
        PATH_TO_IMAGES: path to NIH CXR images
        LABEL: finding of interest (must exactly match one of FINDINGS defined below or will get error)
        PATH_TO_MODEL: path to downloaded pretrained model or your own retrained model
        POSITIVE_FINDINGS_ONLY: dataloader will show only examples + for LABEL pathology if True, otherwise shows positive
                                and negative examples if false

    Returns:
        dataloader: dataloader with test examples to show
        model: fine tuned torchvision densenet-121
    """

    checkpoint = torch.load(PATH_TO_MODEL, map_location=lambda storage, loc: storage)
    model = checkpoint['model']
    del checkpoint
    model = model.module.to(device)
    # model.eval()
    # for param in model.parameters():
    #  param.requires_grad_(False)

    # build dataloader on test
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if not covid:
        bounding_box_transform = CXR.RescaleBB(224, 1024)

        if not POSITIVE_FINDINGS_ONLY:
            finding = "any"
        else:
            finding = LABEL

        dataset = CXR.CXRDataset(
            path_to_images=PATH_TO_IMAGES,
            fold=fold,
            transform=data_transform,
            transform_bb=bounding_box_transform,
            finding=finding)
    else:
        dataset = CXR.CXRDataset(
            path_to_images=PATH_TO_IMAGES,
            fold=fold,
            transform=data_transform,
            fine_tune=True)
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=1)
    
    return iter(dataloader), model


def show_next(cxr, model, label, inputs, filename, bbox):
    """
    Plots CXR, activation map of CXR, and shows model probabilities of findings

    Args:
        dataloader: dataloader of test CXRs
        model: fine-tuned torchvision densenet-121
        LABEL: finding we're interested in seeing heatmap for
    Returns:
        None (plots output)
    """

    raw_cam = calc_cam(inputs, label, model)
    print('range:')
    print(np.ptp(raw_cam))
    print('percerntile:')
    print(np.percentile(raw_cam, 4))
    print('avg:')
    print(np.mean(raw_cam))

    raw_cam = np.array(Image.fromarray(raw_cam.squeeze()).resize((224, 224), Image.NEAREST))

    # bounding box as a mask
    bbox_mask = np.zeros(raw_cam.shape, dtype=bool)
    bbox_mask[bbox[0, 1]: bbox[0, 1] + bbox[0, 3], bbox[0, 0]: bbox[0, 0] + bbox[0, 2]] = True

    bbox_area_ratio = (bbox_mask.sum() / bbox_mask.size) * 100
    activation_mask = np.logical_or(raw_cam >= 180, raw_cam <= 60)
    heat_mask = np.logical_and(raw_cam < 180, raw_cam > 60)

    # finding components in heatmap
    label_im, nb_labels = ndimage.label(activation_mask)
    # print('nb_labels:')
    # print(nb_labels)
    # print('label_im:')
    # print(label_im)

    # heat_mask = label_im == 0
    #
    # components_masks = []
    # for label in range(1, nb_labels + 1):
    #     component_mask = label_im == label
    #     components_masks.append(component_mask)
    object_slices = ndimage.find_objects(label_im)
    detected_patchs = []
    for object_slice in object_slices:
        y_slice = object_slice[0]
        x_slice = object_slice[1]
        xy_corner = (x_slice.start, y_slice.start)
        x_length = x_slice.stop - x_slice.start
        y_length = y_slice.stop - y_slice.start
        detected_patch = patches.Rectangle(xy_corner, x_length, y_length, linewidth=2, edgecolor='m',
                                           facecolor='none', zorder=2)
        detected_patchs.append(detected_patch)

        print(object_slice)

    object_masks = []
    for object_slice in object_slices:
        object_mask = np.zeros(label_im.shape, dtype=bool)
        object_mask[object_slice[0], object_slice[1]] = True
        object_masks.append(object_mask)
    object_masks = np.array(object_masks)

    object_masks_union = np.logical_or.reduce(object_masks)

    def compute_ior(activated_mask, gt_mask):
        intersection_mask = np.logical_and(activated_mask, gt_mask)
        detected_region_area = np.sum(activated_mask)
        # print('detected_area:')
        # print(detected_region_area)
        intersection_area = np.sum(intersection_mask)
        # print('intersection:')
        # print(intersection_area)
        ior = intersection_area / detected_region_area
        return ior

    ior = compute_ior(activation_mask, bbox_mask)
    print('ior:')
    print(ior)
    iobb = compute_ior(object_masks_union, bbox_mask)
    print('iobb:')
    print(iobb)

    fig, (showcxr, heatmap) = plt.subplots(ncols=2, figsize=(14, 5))
    
    hmap = sns.heatmap(raw_cam.squeeze(),
                       cmap='viridis',
                       # vmin= -200, vmax=100,
                       mask=heat_mask,
                       # alpha = 0.8, # whole heatmap is translucent
                       annot=False,
                       zorder=2,
                       linewidths=0)
        
    hmap.imshow(cxr, zorder=1)  # put the map under the heatmap
    hmap.axis('off')
    hmap.set_title('Own Implementation for category {}'.format(label), fontsize=8)

    rect = patches.Rectangle((bbox[0, 0], bbox[0, 1]), bbox[0, 2], bbox[0, 3], linewidth=2, edgecolor='r',
                             facecolor='none', zorder=2)
    hmap.add_patch(rect)

    for patch in detected_patchs:
        hmap.add_patch(patch)

    rect_original = patches.Rectangle((bbox[0, 0], bbox[0, 1]), bbox[0, 2], bbox[0, 3], linewidth=2, edgecolor='r',
                                      facecolor='none', zorder=2)

    showcxr.imshow(cxr)
    showcxr.axis('off')
    showcxr.set_title(filename[0])
    showcxr.add_patch(rect_original)
    # plt.savefig(str(LABEL+"_P"+str(predx[label_index])+"_file_"+filename[0]))
    plt.show()


def eval_localization(dataloader, model, LABEL, map_thresholds, percentiles, ior_threshold=0.1, method='ior'):

    num_correct_pred = 0
    num_images_examined = 0

    def compute_ior(activated_masks, gt_mask):
        intersection_masks = np.logical_and(activated_masks, gt_mask)

        detected_region_areas = np.sum(activated_masks, axis=(1, 2))
        intersection_areas = np.sum(intersection_masks, axis=(1, 2))

        ior = np.divide(intersection_areas, detected_region_areas)

        return ior

    def compute_iou(activated_masks, gt_mask):
        intersection_masks = np.logical_and(activated_masks, gt_mask)
        union_masks = np.logical_or(activated_masks, gt_mask)

        intersection_areas = np.sum(intersection_masks, axis=(1, 2))
        union_areas = np.sum(union_masks, axis=(1, 2))

        iou = np.divide(intersection_areas, union_areas)

        return iou

    map_thresholds = np.array(map_thresholds)
    map_thresholds = map_thresholds[:, np.newaxis, np.newaxis]

    for data in dataloader:

        inputs, labels, filename, bbox = data
        num_images_examined += 1

        # get cam map
        inputs = inputs.to(device)

        raw_cam = calc_cam(inputs, LABEL, model)
        raw_cam = np.array(Image.fromarray(raw_cam.squeeze()).resize((224, 224), Image.NEAREST))

        raw_cams = np.broadcast_to(raw_cam, shape=(len(map_thresholds), raw_cam.shape[0], raw_cam.shape[1]))
        activation_masks = np.greater_equal(raw_cams, map_thresholds)

        # bounding box as a mask
        bbox = bbox.type(torch.cuda.IntTensor)
        bbox_mask = np.zeros(raw_cam.shape, dtype=bool)
        bbox_mask[bbox[0, 1]: bbox[0, 1] + bbox[0, 3], bbox[0, 0]: bbox[0, 0] + bbox[0, 2]] = True

        if method == 'iobb':

            object_masks_union_all_thresholds = []
            for activation_mask in activation_masks:

                label_im, nb_labels = ndimage.label(activation_mask)
                object_slices = ndimage.find_objects(label_im)

                object_masks = []
                for object_slice in object_slices:
                    object_mask = np.zeros(label_im.shape, dtype=bool)
                    object_mask[object_slice[0], object_slice[1]] = True
                    object_masks.append(object_mask)
                object_masks = np.array(object_masks)

                object_masks_union = np.logical_or.reduce(object_masks)
                object_masks_union_all_thresholds.append(object_masks_union)
            object_masks_union_all_thresholds = np.array(object_masks_union_all_thresholds)

            iobb = compute_ior(object_masks_union_all_thresholds, bbox_mask)
            num_correct_pred += np.greater_equal(iobb, ior_threshold)

        if method == 'ior':
            ior = compute_ior(activation_masks, bbox_mask)
            num_correct_pred += np.greater_equal(ior, ior_threshold)

        if method == 'ior_percentile_dynamic':
            bbox_area_ratio = (bbox_mask.sum() / bbox_mask.size) * 100
            activation_mask = raw_cam >= np.percentile(raw_cam, (100 - bbox_area_ratio))
            intersection = np.logical_and(activation_mask, bbox_mask)
            ior = intersection.sum() / activation_mask.sum()
            num_correct_pred += np.greater_equal(ior, ior_threshold)

        if method == 'ior_percentile_static':
            activation_masks = []
            for percentile in percentiles:
                activation_mask = raw_cam >= np.percentile(raw_cam, 100 - percentile)
                activation_masks.append(activation_mask)
            activation_masks = np.array(activation_masks)
            ior = compute_ior(activation_masks, bbox_mask)
            num_correct_pred += np.greater_equal(ior, ior_threshold)

        if method == 'iou':
            iou = compute_iou(activation_masks, bbox_mask)
            num_correct_pred += np.greater_equal(iou, ior_threshold)

        if method == 'iou_percentile_dynamic':
            bbox_area_ratio = (bbox_mask.sum() / bbox_mask.size) * 100
            activation_mask = raw_cam >= np.percentile(raw_cam, (100 - bbox_area_ratio))
            intersection = np.logical_and(activation_mask, bbox_mask)
            union = np.logical_or(activation_mask, bbox_mask)
            iou = intersection.sum() / union.sum()
            num_correct_pred += np.greater_equal(iou, ior_threshold)

        if method == 'iou_percentile_static':
            activation_masks = []
            for percentile in percentiles:
                activation_mask = raw_cam >= np.percentile(raw_cam, 100 - percentile)
                activation_masks.append(activation_mask)
            activation_masks = np.array(activation_masks)
            iou = compute_iou(activation_masks, bbox_mask)
            num_correct_pred += np.greater_equal(iou, ior_threshold)

        if method == 'iou_percentile_bb_dynamic':
            bbox_area_ratio = (bbox_mask.sum() / bbox_mask.size) * 100
            activation_mask = raw_cam >= np.percentile(raw_cam, (100 - bbox_area_ratio))

            label_im, nb_labels = ndimage.label(activation_mask)
            object_slices = ndimage.find_objects(label_im)

            object_masks = []
            for object_slice in object_slices:
                object_mask = np.zeros(label_im.shape, dtype=bool)
                object_mask[object_slice[0], object_slice[1]] = True

                if (np.logical_and(object_mask, bbox_mask)).sum() > 0:
                    object_masks.append(object_mask)

            object_masks = np.array(object_masks)
            object_masks = np.logical_or.reduce(object_masks)

            intersection = np.logical_and(object_masks, bbox_mask)
            union = np.logical_or(object_masks, bbox_mask)
            iou = intersection.sum() / union.sum()
            num_correct_pred += np.greater_equal(iou, ior_threshold)

        if method == 'iou_percentile_bb_dynamic_nih':
            bbox_area_ratio = (bbox_mask.sum() / bbox_mask.size) * 100
            activation_mask = raw_cam >= np.percentile(raw_cam, (100 - bbox_area_ratio))

            label_im, nb_labels = ndimage.label(activation_mask)
            object_slices = ndimage.find_objects(label_im)

            object_masks = []
            for object_slice in object_slices:
                object_mask = np.zeros(label_im.shape, dtype=bool)
                object_mask[object_slice[0], object_slice[1]] = True

                if (np.logical_and(object_mask, bbox_mask)).sum() > 0:
                    object_masks.append(object_mask)

            if len(object_masks) > 0:
                object_masks = np.array(object_masks)
                # object_masks = np.logical_or.reduce(object_masks)

                intersection = np.logical_and(object_masks, bbox_mask)
                union = np.logical_or(object_masks, bbox_mask)
                iou = intersection.sum(axis=(1, 2)) / union.sum(axis=(1, 2))
                iou = np.amax(iou)
                num_correct_pred += np.greater_equal(iou, ior_threshold)

        if method == 'ior_percentile_bb_dynamic_nih':
            bbox_area_ratio = (bbox_mask.sum() / bbox_mask.size) * 100
            activation_mask = raw_cam >= np.percentile(raw_cam, (100 - bbox_area_ratio))

            label_im, nb_labels = ndimage.label(activation_mask)
            object_slices = ndimage.find_objects(label_im)

            object_masks = []
            for object_slice in object_slices:
                object_mask = np.zeros(label_im.shape, dtype=bool)
                object_mask[object_slice[0], object_slice[1]] = True

                if (np.logical_and(object_mask, bbox_mask)).sum() > 0:
                    object_masks.append(object_mask)

            if len(object_masks) > 0:
                object_masks = np.array(object_masks)
                # object_masks = np.logical_or.reduce(object_masks)

                intersection = np.logical_and(object_masks, bbox_mask)
                # union = np.logical_or(object_masks, bbox_mask)
                iou = intersection.sum(axis=(1, 2)) / object_masks.sum(axis=(1, 2))
                iou = np.amax(iou)
                num_correct_pred += np.greater_equal(iou, ior_threshold)

    accuracy = num_correct_pred / num_images_examined
    return accuracy


def plot_map(model, dataloader, label=None, covid=False, saliency_layer=None):
    """Plot an example.

    Args:
        model: trained classification model
        dataloader: containing input images.
        label (str): Name of Category.
        covid: whether the image is from the Covid Dataset or the Chesxtray Dataset.
        saliency_layer: usually output of the last convolutional layer.
    """

    if not covid:
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
    else:
        FINDINGS = [
            'Detector01',
            'Detector2',
            'Detector3']

    try:
        if not covid:
            inputs, labels, filename, bbox = next(dataloader)
            bbox = bbox.type(torch.cuda.IntTensor)
        else:
            inputs, labels, filename = next(dataloader)
    except StopIteration:
        print("All examples exhausted - rerun cells above to generate new examples to review")
        return None

    original = inputs.clone()
    inputs = inputs.to(device)
    original = original.to(device)
    original.requires_grad = True

    # create predictions for label of interest and all labels
    pred = torch.sigmoid(model(original)).data.cpu().numpy()[0]
    predx = ['%.3f' % elem for elem in list(pred)]

    preds_concat = pd.concat([pd.Series(FINDINGS), pd.Series(predx), pd.Series(labels.numpy().astype(bool)[0])], axis=1)
    preds = pd.DataFrame(data=preds_concat)
    preds.columns = ["Finding", "Predicted Probability", "Ground Truth"]
    preds.set_index("Finding", inplace=True)
    preds.sort_values(by='Predicted Probability', inplace=True, ascending=False)

    cxr = inputs.data.cpu().numpy().squeeze().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    cxr = std * cxr + mean
    cxr = np.clip(cxr, 0, 1)

    if not covid:
        show_next(cxr, model, label, inputs, filename, bbox)

    if covid and label is None:
        label = preds.loc[preds["Ground Truth"]==True].index[0]

    category_id = FINDINGS.index(label)

    saliency = grad_cam(model, original, category_id, saliency_layer=saliency_layer)

    fig, (showcxr, heatmap) = plt.subplots(ncols=2, figsize=(14, 5))

    showcxr.imshow(cxr)
    showcxr.axis('off')
    showcxr.set_title(filename[0])
    if not covid:
        rect_original = patches.Rectangle((bbox[0, 0], bbox[0, 1]), bbox[0, 2], bbox[0, 3], linewidth=2, edgecolor='r',
                                          facecolor='none', zorder=2)
        showcxr.add_patch(rect_original)

    hmap = sns.heatmap(saliency.detach().cpu().numpy().squeeze(),
                       cmap='viridis',
                       annot=False,
                       zorder=2,
                       linewidths=0)
    hmap.axis('off')
    hmap.set_title('TorchRay grad cam for category {}'.format(label), fontsize=8)

    plt.show()

    print(preds)

    if covid:
        data_brixia = pd.read_csv("model/labels/metadata_global_v2.csv", sep=";")
        data_brixia.set_index("Filename", inplace=True)
        score = data_brixia.loc[filename[0].replace(".jpg", ".dcm"), "BrixiaScore"].astype(str)
        print('Brixia 6 regions Score: ', '0' * (6 - len(score)) + score)
