import pandas as pd
import numpy as np

# Script for using the csv file on the original NIH ChestXray dataset and it's corresponding test and train_val
# lists by bringing them all into one dataframe and changing the labeling into a one hot scheme (Like the csv
# file provided on CheXnet repository)

data_nih_all_original = pd.read_csv('labels/Data_Entry_2017.csv')
data_nih_test = pd.read_csv('labels/test_list.txt', header=None, names=['Image Index'])
data_nih_bbox = pd.read_csv("labels/BBox_List_2017.csv")
data_train_val = pd.read_csv('labels/train_val_list.txt', header=None, names=['Image Index'])
# data_chexnet_split = pd.read_csv("nih_labels.csv")

# droping the extra columns that we do not use (such as Age, ...)
data_nih_all_original.drop(data_nih_all_original.columns[2:], axis=1, inplace=True)

labels = [
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

# Adding a column for each of the labels (In order to create a one-hot structure)
len_nih = len(data_nih_all_original['Image Index'])
for label in labels:
    data_nih_all_original[label] = pd.Series(
        np.zeros((len_nih,), dtype=int), index=data_nih_all_original.index, dtype='int')


# helper function for use in the df.apply
def label_exists(label, finding):
    if label in finding:
        return 1
    else:
        return 0


# create the one hot encoding by checking the 'Finding labels' columns
for label in labels:
    data_nih_all_original[label] = data_nih_all_original.apply(
        lambda row: label_exists(label, row['Finding Labels']), axis=1)

data_nih = data_nih_all_original

# Do the split on train_val list
data_train_val['msk'] = np.random.rand(len(data_train_val['Image Index'])) < 0.88
data_train_val.loc[data_train_val.msk == True, 'fold'] = 'train'
data_train_val.loc[data_train_val.msk == False, 'fold'] = 'val'
data_train_val.drop(['msk'], axis=1, inplace=True)

# adding the train and validation labels to the original dataframe
data_nih = pd.merge(data_nih, data_train_val, how='left', on='Image Index')

# marking the test data in the original data set
data_nih_test['fold'] = 'test'
data_nih = pd.merge(data_nih, data_nih_test, how='left', on='Image Index')

# bringing all the info about the split under one column, 'fold'
data_nih.loc[data_nih.fold_y == 'test', 'fold_x'] = 'test'
data_nih.drop(['fold_y'], axis=1, inplace=True)

data_nih = data_nih.rename(columns={'fold_x': 'fold'})

print(data_nih_test.shape)
print(data_nih.shape)
print(data_train_val.shape)
print(data_nih[data_nih.fold == 'train'].shape)
print(data_nih[data_nih.fold == 'val'].shape)
print(data_nih[data_nih.fold == 'test'].shape)

data_nih.to_csv('labels/nih_original_split.csv', index=False)
