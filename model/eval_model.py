import torch
import pandas as pd
import cxr_dataset as CXR
from torch.utils.data import Dataset, DataLoader
import sklearn.metrics as sklm
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_pred_multilabel(dataloader, model, save_as_csv=False, fine_tune=False):
    """
    Gives predictions for test fold and calculates AUCs using previously trained model

    Args:
        data_transforms: torchvision transforms to preprocess raw images; same as validation transforms
        model: densenet-121 from torchvision previously fine tuned to training data
        PATH_TO_IMAGES: path at which NIH images can be found
    Returns:
        pred_df: dataframe containing individual predictions and ground truth for each test image
        auc_df: dataframe containing aggregate AUCs by train/test tuples
    """

    batch_size = dataloader.batch_size
    # set model to eval mode; required for proper predictions given use of batchnorm
    model.train(False)

    # create empty dfs
    pred_df = pd.DataFrame(columns=["Image Index"])
    true_df = pd.DataFrame(columns=["Image Index"])

    # iterate over dataloader
    for i, data in enumerate(dataloader):

        inputs, labels, _ = data
        inputs, labels = inputs.to(device), labels.to(device)

        true_labels = labels.cpu().data.numpy()
        # batch_size = true_labels.shape

        outputs = model(inputs)
        outputs = torch.sigmoid(outputs)
        probs = outputs.cpu().data.numpy()

        # get predictions and true values for each item in batch
        for j in range(0, true_labels.shape[0]):
            thisrow = {}
            truerow = {}
            thisrow["Image Index"] = dataloader.dataset.df.index[batch_size * i + j]
            truerow["Image Index"] = dataloader.dataset.df.index[batch_size * i + j]

            # iterate over each entry in prediction vector; each corresponds to
            # individual label
            for k in range(len(dataloader.dataset.PRED_LABEL)):
                thisrow["prob_" + dataloader.dataset.PRED_LABEL[k]] = probs[j, k]
                truerow[dataloader.dataset.PRED_LABEL[k]] = true_labels[j, k]

            pred_df = pred_df.append(thisrow, ignore_index=True)
            true_df = true_df.append(truerow, ignore_index=True)

        # if(i % 10 == 0):
        #     print(str(i * BATCH_SIZE))

    auc_df = pd.DataFrame(columns=["label", "auc"])

    # calc AUCs
    for column in true_df:

        if not fine_tune:
            if column not in [
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
                    'Hernia']:
                        continue
        else:
            if column not in [
                'Detector01',
                'Detector2',
                    'Detector3']:
                        continue
        actual = true_df[column]
        pred = pred_df["prob_" + column]
        thisrow = {}
        thisrow['label'] = column
        thisrow['auc'] = np.nan
        thisrow['AP'] = np.nan
        try:
            thisrow['auc'] = sklm.roc_auc_score(actual.to_numpy().astype(int), pred.to_numpy())
            thisrow['AP'] = sklm.average_precision_score(actual.to_numpy().astype(int), pred.to_numpy())
        except BaseException:
            print("can't calculate auc for " + str(column))
        auc_df = auc_df.append(thisrow, ignore_index=True)

    if save_as_csv:
        pred_df.to_csv("results/preds.csv", index=False)
        auc_df.to_csv("results/aucs.csv", index=False)

    return pred_df, auc_df


def evaluate_mae(dataloader, model):
    """
    Calculates MAE using previously trained model

    Args:
        data_transforms: torchvision transforms to preprocess raw images; same as validation transforms
        model: densenet-121 from torchvision previously fine tuned to training data
    Returns:
        mae: MAE
    """

    # calc preds in batches of 32, can reduce if your GPU has less RAM
    batch_size = dataloader.batch_size
    # set model to eval mode; required for proper predictions given use of batchnorm
    model.train(False)

    # create empty dfs
    pred_df = pd.DataFrame(columns=["Image Index"])
    true_df = pd.DataFrame(columns=["Image Index"])

    # iterate over dataloader
    for i, data in enumerate(dataloader):

        inputs, ground_truths, _ = data
        inputs, ground_truths = inputs.to(device), ground_truths.to(device)

        true_scores = ground_truths.cpu().data.numpy()

        outputs = model(inputs)
        preds = outputs.cpu().data.numpy()

        # get predictions and true values for each item in batch
        for j in range(0, true_scores.shape[0]):
            thisrow = {}
            truerow = {}
            thisrow["Image Index"] = dataloader.dataset.df.index[batch_size * i + j]
            truerow["Image Index"] = dataloader.dataset.df.index[batch_size * i + j]

            # iterate over each entry in prediction vector; each corresponds to
            # individual label
            thisrow["pred_score"] = preds[j]
            truerow["true_score"] = true_scores[j]

            pred_df = pred_df.append(thisrow, ignore_index=True)
            true_df = true_df.append(truerow, ignore_index=True)

    actual = true_df["true_score"]
    pred = pred_df["pred_score"]
    try:
        mae = sklm.mean_absolute_error(actual.to_numpy().astype(int), pred.to_numpy())
        return mae, true_df, pred_df
    except BaseException:
        print("can't calculate mae")

