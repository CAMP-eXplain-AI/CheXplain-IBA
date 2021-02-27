from __future__ import print_function, division

# pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torchvision import transforms, utils
from tensorboardX import SummaryWriter

# general imports
import os
import time
from shutil import rmtree

# data science imports
import csv

import cxr_dataset as CXR
import eval_model as E

use_gpu = torch.cuda.is_available()
gpu_count = torch.cuda.device_count()
print("Available GPU count:" + str(gpu_count))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def checkpoint(model, best_loss, epoch, LR, filename):
    """
    Saves checkpoint of torchvision model during training.

    Args:
        model: torchvision model to be saved
        best_loss: best val loss achieved so far in training
        epoch: current epoch of training
        LR: current learning rate in training
    Returns:
        None
    """

    print('saving')
    state = {
        'model': model,
        'best_loss': best_loss,
        'epoch': epoch,
        'rng_state': torch.get_rng_state(),
        'LR': LR
    }

    torch.save(state, 'results/' + filename)


def pos_neg_weights_in_batch(labels_batch):

    num_total = labels_batch.shape[0] * labels_batch.shape[1]
    num_positives = labels_batch.sum()
    num_negatives = num_total - num_positives

    if not num_positives == 0:
        beta_p = num_negatives / num_positives
    else:
        beta_p = num_negatives
    beta_p = torch.as_tensor(beta_p)
    beta_p = beta_p.to(device)
    beta_p = beta_p.type(torch.cuda.FloatTensor)

    return beta_p


def train_model(
        model,
        criterion,
        optimizer,
        LR,
        num_epochs,
        dataloaders,
        dataset_sizes,
        weight_decay,
        weighted_cross_entropy_batchwise=False,
        fine_tune=False,
        regression=False):
    """
    Fine tunes torchvision model to NIH CXR data.

    Args:
        model: torchvision model to be finetuned (densenet-121 in this case)
        criterion: loss criterion (binary cross entropy loss, BCELoss)
        optimizer: optimizer to use in training (SGD)
        LR: learning rate
        num_epochs: continue training up to this many epochs
        dataloaders: pytorch train and val dataloaders
        dataset_sizes: length of train and val datasets
        weight_decay: weight decay parameter we use in SGD with momentum
    Returns:
        model: trained torchvision model
        best_epoch: epoch on which best model val loss was obtained
    """
    since = time.time()

    start_epoch = 1
    best_loss = 999999
    best_epoch = -1
    last_train_loss = -1

    tensorboard_writer_train = SummaryWriter('runs/loss/train_loss')
    tensorboard_writer_val = SummaryWriter('runs/loss/val_loss')

    if not fine_tune:
        PRED_LABEL = [
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
        PRED_LABEL = [
            'Detector01',
            'Detector2',
            'Detector3']

    if not regression:
        tensorboard_writer_auc = {}
        tensorboard_writer_AP = {}
        for label in PRED_LABEL:
            tensorboard_writer_auc[label] = SummaryWriter('runs/auc/'+label)
            tensorboard_writer_AP[label] = SummaryWriter('runs/ap/' + label)
    else:
        tensorboard_writer_mae = SummaryWriter('runs/mae')

    # iterate over epochs
    for epoch in range(start_epoch, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # set model to train or eval mode based on whether we are in train or
        # val; necessary to get correct predictions given batchnorm
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0

            total_done = 0

            for data in dataloaders[phase]:
                if not regression:
                    inputs, labels, _ = data
                else:
                    inputs, ground_truths, _ = data
                batch_size = inputs.shape[0]
                inputs = inputs.to(device)
                if not regression:
                    labels = (labels.to(device)).float()
                else:
                    ground_truths = (ground_truths.to(device)).float()

                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)

                    # calculate gradient and update parameters in train phase
                    optimizer.zero_grad()

                    if weighted_cross_entropy_batchwise:
                        beta = pos_neg_weights_in_batch(labels)
                        criterion = nn.BCEWithLogitsLoss(pos_weight=beta)

                    if not regression:
                        loss = criterion(outputs, labels)
                    else:
                        ground_truths = ground_truths.unsqueeze(1)
                        loss = criterion(outputs, ground_truths)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * batch_size

            epoch_loss = running_loss / dataset_sizes[phase]

            if phase == 'train':
                tensorboard_writer_train.add_scalar('Loss', epoch_loss, epoch)
                last_train_loss = epoch_loss
            elif phase == 'val':
                tensorboard_writer_val.add_scalar('Loss', epoch_loss, epoch)

                if not regression:
                    preds, aucs = E.make_pred_multilabel(dataloaders['val'], model, save_as_csv=False, fine_tune=fine_tune)
                    aucs.set_index('label', inplace=True)
                    print(aucs)
                    for label in PRED_LABEL:
                        tensorboard_writer_auc[label].add_scalar('AUC', aucs.loc[label, 'auc'], epoch)
                        tensorboard_writer_AP[label].add_scalar('AP', aucs.loc[label, 'AP'], epoch)
                else:
                    mae, _, _ = E.evaluate_mae(dataloaders['val'], model)
                    print('MAE: ', mae)
                    tensorboard_writer_mae.add_scalar('MAE', mae, epoch)

            print(phase + ' epoch {}:loss {:.4f} with data size {}'.format(
                epoch, epoch_loss, dataset_sizes[phase]))

            # checkpoint model if has best val loss yet
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch
                if not fine_tune:
                    checkpoint(model, best_loss, epoch, LR, filename='checkpoint_best')
                elif fine_tune and not regression:
                    checkpoint(model, best_loss, epoch, LR, filename='classification_checkpoint_best')
                else:
                    checkpoint(model, best_loss, epoch, LR, filename='regression_checkpoint_best')

        # log training and validation loss over each epoch
        with open("results/log_train", 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            if epoch == 1:
                logwriter.writerow(["epoch", "train_loss", "val_loss"])
            logwriter.writerow([epoch, last_train_loss, epoch_loss])

        # Save model after each epoch
        # checkpoint(model, best_loss, epoch, LR, filename='checkpoint')

        total_done += batch_size
        if total_done % (100 * batch_size) == 0:
            print("completed " + str(total_done) + " so far in epoch")

        # print elapsed time from the beginning after each epoch
        print('Training complete in {:.0f}m {:.0f}s'.format(
            (time.time() - since) // 60, (time.time() - since) % 60))

    # total time
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights to return
    if not fine_tune:
        checkpoint_best = torch.load('results/checkpoint_best')
    elif fine_tune and not regression:
        checkpoint_best = torch.load('results/classification_checkpoint_best')
    else:
        checkpoint_best = torch.load('results/regression_checkpoint_best')
    model = checkpoint_best['model']
    return model, best_epoch


def train_cnn(PATH_TO_IMAGES, LR, WEIGHT_DECAY, fine_tune=False, regression=False, freeze=False, adam=False,
              initial_model_path=None, initial_brixia_model_path=None, weighted_cross_entropy_batchwise=False,
              modification=None, weighted_cross_entropy=False):
    """
    Train torchvision model to NIH data given high level hyperparameters.

    Args:
        PATH_TO_IMAGES: path to NIH images
        LR: learning rate
        WEIGHT_DECAY: weight decay parameter for SGD

    Returns:
        preds: torchvision model predictions on test fold with ground truth for comparison
        aucs: AUCs for each train,test tuple

    """
    NUM_EPOCHS = 100
    BATCH_SIZE = 32

    try:
        rmtree('results/')
    except BaseException:
        pass  # directory doesn't yet exist, no need to clear it
    os.makedirs("results/")

    # use imagenet mean,std for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    N_LABELS = 14  # we are predicting 14 labels
    N_COVID_LABELS = 3  # we are predicting 3 COVID labels

    # define torchvision transforms
    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }

    # create train/val dataloaders
    transformed_datasets = {}
    transformed_datasets['train'] = CXR.CXRDataset(
        path_to_images=PATH_TO_IMAGES,
        fold='train',
        transform=data_transforms['train'],
        fine_tune=fine_tune,
        regression=regression)
    transformed_datasets['val'] = CXR.CXRDataset(
        path_to_images=PATH_TO_IMAGES,
        fold='val',
        transform=data_transforms['val'],
        fine_tune=fine_tune,
        regression=regression)

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(
        transformed_datasets['train'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8)
    dataloaders['val'] = torch.utils.data.DataLoader(
        transformed_datasets['val'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8)

    # please do not attempt to train without GPU as will take excessively long
    if not use_gpu:
        raise ValueError("Error, requires GPU")

    if initial_model_path or initial_brixia_model_path:
        if initial_model_path:
            saved_model = torch.load(initial_model_path)
        else:
            saved_model = torch.load(initial_brixia_model_path)
        model = saved_model['model']
        del saved_model
        if fine_tune and not initial_brixia_model_path:
            num_ftrs = model.module.classifier.in_features
            if freeze:
                for feature in model.module.features:
                    for param in feature.parameters():
                        param.requires_grad = False
                    if feature == model.module.features.transition2:
                        break
            if not regression:
                model.module.classifier = nn.Linear(num_ftrs, N_COVID_LABELS)
            else:
                model.module.classifier = nn.Sequential(
                    nn.Linear(num_ftrs, 1),
                    nn.ReLU(inplace=True)
                )
    else:
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, N_LABELS)

        if modification == 'transition_layer':
            # num_ftrs = model.features.norm5.num_features
            up1 = torch.nn.Sequential(torch.nn.ConvTranspose2d(num_ftrs, num_ftrs, kernel_size=3, stride=2, padding=1),
                                      torch.nn.BatchNorm2d(num_ftrs),
                                      torch.nn.ReLU(True))
            up2 = torch.nn.Sequential(torch.nn.ConvTranspose2d(num_ftrs, num_ftrs, kernel_size=3, stride=2, padding=1),
                                      torch.nn.BatchNorm2d(num_ftrs))

            transition_layer = torch.nn.Sequential(up1, up2)
            model.features.add_module('transition_chestX', transition_layer)

        if modification == 'remove_last_block':
            model.features.denseblock4 = nn.Sequential()
            model.features.transition3 = nn.Sequential()
            # model.features.norm5 = nn.BatchNorm2d(512)
            # model.classifier = nn.Linear(512, N_LABELS)
        if modification == 'remove_last_two_block':
            model.features.denseblock4 = nn.Sequential()
            model.features.transition3 = nn.Sequential()

            model.features.transition2 = nn.Sequential()
            model.features.denseblock3 = nn.Sequential()

            model.features.norm5 = nn.BatchNorm2d(512)
            model.classifier = nn.Linear(512, N_LABELS)

    print(model)

    # put model on GPU
    if not initial_model_path:
        model = nn.DataParallel(model)
    model.to(device)

    if regression:
        criterion = nn.MSELoss()
    else:
        if weighted_cross_entropy:
            pos_weights = transformed_datasets['train'].pos_neg_balance_weights()
            print(pos_weights)
            # pos_weights[pos_weights>40] = 40
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        else:
            criterion = nn.BCEWithLogitsLoss()

    if adam:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY, momentum=0.9)

    dataset_sizes = {x: len(transformed_datasets[x]) for x in ['train', 'val']}

    # train model
    if regression:
        model, best_epoch = train_model(model, criterion, optimizer, LR, num_epochs=NUM_EPOCHS,
                                        dataloaders=dataloaders, dataset_sizes=dataset_sizes,
                                        weight_decay=WEIGHT_DECAY, fine_tune=fine_tune, regression=regression)
    else:
        model, best_epoch = train_model(model, criterion, optimizer, LR, num_epochs=NUM_EPOCHS,
                                        dataloaders=dataloaders, dataset_sizes=dataset_sizes, weight_decay=WEIGHT_DECAY,
                                        weighted_cross_entropy_batchwise=weighted_cross_entropy_batchwise,
                                        fine_tune=fine_tune)
        # get preds and AUCs on test fold
        preds, aucs = E.make_pred_multilabel(dataloaders['val'], model, save_as_csv=False, fine_tune=fine_tune)
        return preds, aucs
