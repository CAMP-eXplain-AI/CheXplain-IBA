from __future__ import print_function, division

# pytorch imports
import torch
from torchray.utils import imsc
from torchray.attribution.grad_cam import grad_cam
from torchray.attribution.gradient import gradient as torchray_gradient
from torchray.attribution.deconvnet import deconvnet
from torchray.attribution.excitation_backprop import excitation_backprop
from torchray.attribution.guided_backprop import guided_backprop
from torchray.attribution.linear_approx import linear_approx
from torchvision import transforms

# image / graphics imports
from PIL import Image
from pylab import *
import seaborn as sns
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from skimage.transform import resize

# data science
import numpy as np
import pandas as pd
from scipy import ndimage

# import other modules
import cxr_dataset as CXR


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data(
        PATH_TO_IMAGES,
        LABEL,
        PATH_TO_MODEL,
        fold,
        POSITIVE_FINDINGS_ONLY=None,
        covid=False,
        regression=False,
        label_path=None):
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
            finding=finding,
            label_path=label_path)
    else:        
        dataset = CXR.CXRDataset(
            path_to_images=PATH_TO_IMAGES,
            fold=fold,
            transform=data_transform,
            fine_tune=True,
            regression=regression,
            label_path=label_path)
    
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
    activation_mask = np.logical_or(raw_cam >= 180 , raw_cam <= 60)
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

def to_saliency_map(saliency_map, shape=None):
    """
    Converts the layer capacity (in nats) to a saliency map (in bits) of the given shape.
    PyTorch:    Use data_format == 'channels_first'
    Tensorflow: Use data_format == 'channels_last'
    """
    if shape is not None:
        saliency_map
        ho, wo = saliency_map.shape
        h, w = shape
        # Scale bits to the pixels
        saliency_map *= (ho*wo) / (h*w)
        return resize(saliency_map, shape, order=1, preserve_range=True)
    else:
        return saliency_map

def plot_saliency_map(saliency_map, img=None, ax=None,
                      colorbar=True,
                      colorbar_label='Bits / Pixel',
                      colorbar_fontsize=14,
                      min_alpha=0.2, max_alpha=0.7, vmax=None,
                      colorbar_size=0.3, colorbar_pad=0.08):

    """
    Plots the heatmap with an bits/pixel colorbar and optionally overlays the image.

    Args:
        saliency_map (np.ndarray): the saliency_map.
        img (np.ndarray):  show this image under the saliency_map.
        ax: matplotlib axis. If ``None``, a new plot is created.
        colorbar_label (str): label for the colorbar.
        colorbar_fontsize (int): fontsize of the colorbar label.
        min_alpha (float): minimum alpha value for the overlay. only used if ``img`` is given.
        max_alpha (float): maximum alpha value for the overlay. only used if ``img`` is given.
        vmax: maximum value for colorbar.
        colorbar_size: width of the colorbar. default: Fixed(0.3).
        colorbar_pad: width of the colorbar. default: Fixed(0.08).

    Returns:
        The matplotlib axis ``ax``.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    from mpl_toolkits.axes_grid1.axes_size import Fixed
    from skimage.color import rgb2grey, grey2rgb
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.0))

    if img is not None:
        # Underlay the image as greyscale
        grey = grey2rgb(rgb2grey(img))
        ax.imshow(grey)

    ax1_divider = make_axes_locatable(ax)
    if type(colorbar_size) == float:
        colorbar_size = Fixed(colorbar_size)
    if type(colorbar_pad) == float:
        colorbar_pad = Fixed(colorbar_pad)
    if vmax is None:
        vmax = saliency_map.max()
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
    n = 256
    half_jet_rgba = plt.cm.seismic(np.linspace(0.5, 1, n))
    half_jet_rgba[:, -1] = np.linspace(0.2, 1, n)
    cmap = mpl.colors.ListedColormap(half_jet_rgba)
    hmap_jet = cmap(norm(saliency_map))
    if img is not None:
        hmap_jet[:, :, -1] = (max_alpha - min_alpha)*norm(saliency_map) + min_alpha
    ax.imshow(hmap_jet, alpha=max_alpha)
    if colorbar:
      cax1 = ax1_divider.append_axes("right", size=colorbar_size, pad=colorbar_pad)
      cbar = mpl.colorbar.ColorbarBase(cax1, cmap=cmap, norm=norm)
      cbar.set_label(colorbar_label, fontsize=colorbar_fontsize)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid('off')
    ax.set_frame_on(False)
    return ax

class Plotter:
  def __init__(self,
  target=None,
  dev=None,
  informationbottleneck=None,
  iba_loss_function=None,
  model=None,
  local_score_dataset=None):
    self.target = target
    self.dev = dev
    self.informationbottleneck = informationbottleneck
    self.iba_loss_function = iba_loss_function
    self.model = model
    if local_score_dataset is not None:
      self.local_score_dataset = local_score_dataset

  def plot_map(self, model, dataloader, label, methods=None, covid=False, regression=False, saliency_layer=None, overlay=False, axes_a=None):
      """Plot an example.

      Args:
          model: trained classification model
          dataloader: containing input images.
          label (str): Name of Category.
          methods: list of attribution methods
          covid: whether the image is from the Covid Dataset or the Chesxtray Dataset.
          saliency_layer: usually output of the last convolutional layer.
          overlay: if display saliency map over image
          axes_a: axis of plot
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
              'NoCovid',
              'Covid']

      try:
          if not covid:
              inputs, labels, filename, bbox = next(dataloader)
              bbox = bbox.type(torch.cuda.IntTensor)
          else:
              inputs, labels, filename = next(dataloader)
              # for consistant return value
              bbox = None
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

      # normalize image
      cxr = inputs.data.cpu().numpy().squeeze().transpose(1, 2, 0)
      mean = np.array([0.485, 0.456, 0.406])
      std = np.array([0.229, 0.224, 0.225])
      cxr = std * cxr + mean
      cxr = np.clip(cxr, 0, 1)

      # In case we want to visualize COVID, we use the highest probability as label for the visualization
      if covid and label is None:
          label = preds.loc[preds["Ground Truth"] == True].index[0]

      category_id = FINDINGS.index(label)

      # if not covid:
      #     show_iba(cxr, model, label, inputs, labels, filename, bbox)
      #     show_iba_new(cxr, model, label, inputs, labels, filename, bbox)
      #      show_next(cxr, model, label, inputs, filename, bbox)

      if methods is None:
        methods = ['grad-cam backprop', 'gradient', 'deconvnet', 'excitation backprop', 'guided backprop', 'linear approx', 'original IB', 'IB with reversed mask']

      # plot original data with bounding box, else show brixia score
      showcxr = axes_a.flatten()[0]
      showcxr.imshow(cxr)
      showcxr.axis('off')
      showcxr.set_title(filename[0])
      if not covid:
          rect_original = patches.Rectangle((bbox[0, 0], bbox[0, 1]), bbox[0, 2], bbox[0, 3], linewidth=2, edgecolor='r',
                                            facecolor='none', zorder=2)
          showcxr.add_patch(rect_original)
      else:
            scores = self.local_score_dataset.getScore(filename[0])
            color_list = ["green", "yellow", "red", "black"]
            for idx, score in enumerate(scores):
              row = (1-idx%3/2)*0.8 + 0.1
              col = idx//3 * 0.8 + 0.1
              plt.text(col, row, score, 
                      color="white",
                      fontsize = 36,
                      bbox=dict(facecolor=color_list[score], alpha=0.7), 
                      transform=showcxr.transAxes)

      # plot visulizations
      for method, hmap in zip(methods, axes_a.flatten()[1:]):
          if method == 'grad-cam backprop':
              saliency = grad_cam(model, original, category_id, saliency_layer=saliency_layer)
          elif method == 'gradient':
              saliency = torchray_gradient(model, original, category_id)
          elif method == 'deconvnet':
              saliency = deconvnet(model, original, category_id)
          elif method == 'excitation backprop':
              saliency = excitation_backprop(model, original, category_id, saliency_layer=saliency_layer)
          elif method == 'guided backprop':
              saliency = guided_backprop(model, original, category_id)
          elif method == 'linear approx':
              saliency = linear_approx(model, original, category_id, saliency_layer=saliency_layer)
          elif method == 'original IB':
              saliency = self.iba(original, labels.squeeze(), model_loss_closure_with_target=self.softmax_crossentropy_loss_with_target)
          elif method == 'IB with reversed mask':
              saliency = self.iba(original, labels.squeeze(), reverse_lambda=True, model_loss_closure_with_target=self.softmax_crossentropy_loss_with_target)

          if not overlay:
            sns.heatmap(saliency.detach().cpu().numpy().squeeze(),
                              cmap='viridis',
                              annot=False,
                              square=True,
                              cbar=False,
                              zorder=2,
                              linewidths=0,
                              ax=hmap)
          else:
            # resize saliancy map
            if type(saliency).__module__ == np.__name__:
              np_saliency = saliency
            else:
              np_saliency = saliency.detach().cpu().numpy().squeeze()
            np_saliency = to_saliency_map(np_saliency, cxr.shape[:2])

            # plot saliency map with image
            plot_saliency_map(np_saliency, cxr, ax = hmap, colorbar=False)
          hmap.axis('off')
          hmap.set_title('{} for category {}'.format(method, label), fontsize=12)

      return inputs, labels, filename, bbox, preds

  def iba(self, img, target, reverse_lambda=False, model_loss_closure_with_target=None):
    """
    Plots CXR, attribution map of CXR generated by IB
    Args:
      img: image
      target ((int)): target label
      reverse lambda: whether use reversed lambda
    """

    img = img[None].to(self.dev)
    self.informationbottleneck.reverse_lambda = reverse_lambda
    heatmap = self.informationbottleneck.analyze(img.squeeze(0), model_loss_closure_with_target(target))

    return heatmap

  def softmax_crossentropy_loss_with_target(self, target):
    def model_loss_closure(input):
      loss = torch.nn.BCEWithLogitsLoss()
      mse_loss = loss(self.model(input), torch.tensor(target).view(1,-1).expand(10, -1).to(self.dev).float())
      return mse_loss
    return model_loss_closure

  def mse_loss_with_target(self, target):
    def model_loss_closure(input):
      loss = torch.nn.MSELoss()
      mse_loss = loss(model(input), torch.tensor(target).to(self.dev))
      return mse_loss
    return model_loss_closure

  def mse_loss_maximize_score(self, target=None):
    def model_loss_closure(input):
      loss = torch.nn.MSELoss()
      mse_loss = -loss(model(input), torch.tensor(0.).to(self.dev))
      return mse_loss
    return model_loss_closure
  
  def mse_loss_minimal_deviation(self, target=None):
    def model_loss_closure(input):
      loss = torch.nn.MSELoss()
      mse_loss = loss(model(input), original_model(input))
      return mse_loss
    return model_loss_closure
