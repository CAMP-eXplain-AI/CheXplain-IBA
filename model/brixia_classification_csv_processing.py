import pandas as pd
from sklearn.model_selection import train_test_split

data_brixia = pd.read_csv("model/labels/metadata_global_v2.csv", sep=";")

# droping the extra columns that we do not use (such as Age, ...)
data_brixia = data_brixia[["Filename", "BrixiaScoreGlobal", "BrixiaScore"]]
data_brixia.rename(columns={"Filename": "Image Index"}, inplace=True)

data_brixia["Image Index"] = data_brixia["Image Index"].apply(lambda x: x.replace(".dcm", ".jpg"))

data_brixia["BrixiaScoreBinary"] = data_brixia["BrixiaScore"].apply(lambda x: int(('2' in str(x)) or ('3' in str(x))))

# Do the split on train_val list
data_train_test, data_val = train_test_split(data_brixia, test_size=0.12)
data_train, data_test = train_test_split(data_train_test, test_size=0.05)
data_train['fold'] = 'train'
data_val['fold'] = 'val'
data_test['fold'] = 'test'

data_brixia = data_train.append(data_val)
data_brixia = data_brixia.append(data_test).sort_index()

data_brixia.loc[data_brixia["BrixiaScoreGlobal"].isin(range(1, 7)), "BrixiaScoreGlobal"] = 1
data_brixia.loc[data_brixia["BrixiaScoreGlobal"].isin(range(7, 13)), "BrixiaScoreGlobal"] = 2
data_brixia.loc[data_brixia["BrixiaScoreGlobal"].isin(range(13, 19)), "BrixiaScoreGlobal"] = 3

data_brixia.to_csv('model/labels/brixia_split_classification.csv', index=False)

