import yaml
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import requests
from sktime.transformations.panel.rocket import Rocket
from sklearn.utils.validation import check_array
from sktime.datatypes._panel._convert import from_2d_array_to_nested

# Ottieni l'argomento dalla riga di comando
dataset = sys.argv[1]
rocket = sys.argv[2]
rocket = bool(rocket)
kernels = sys.argv[3]
kernels = int(kernels)
seed = sys.argv[4]
print("Argument: ", dataset)

class PrettySafeLoader(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))

PrettySafeLoader.add_constructor(
    u'tag:yaml.org,2002:python/tuple',
    PrettySafeLoader.construct_python_tuple)

print("Reading dataset info...")
# Leggi il dataset e ottieni le forme di x_train e y_train
train_df = pd.read_table("../../UCRArchive_2018/" + dataset + "/" + dataset + "_TRAIN.tsv", header=None)
train_df = train_df.rename(columns={0: 'class'})
test_df = pd.read_table("../../UCRArchive_2018/" + dataset + "/" + dataset + "_TEST.tsv", header=None)
test_df = test_df.rename(columns={0: 'class'})

if train_df.isna().any().any():
    train_df = train_df.dropna(axis = 1, how='any') #drop NaN by columns
    train_df = train_df.dropna(axis = 0, how='any') #drop NaN by rows


test_df = test_df[train_df.columns.intersection(test_df.columns)] #now test set will have the same columns of train set after removing columns
if test_df.isna().any().any():
    test_df = test_df.dropna(axis = 0, how='any') #remove missing values by rows, because columns must be the same of train set

# Normalization
scaler = StandardScaler()
train_df.iloc[: , 1:] = scaler.fit_transform(train_df.iloc[: , 1:])
test_df.iloc[: , 1:] = scaler.transform(test_df.iloc[: , 1:])

if len(train_df["class"].unique()) == 2:
    train_df[train_df["class"] == -1] = 0
if len(test_df["class"].unique()) == 2:
    test_df[test_df["class"] == -1] = 0

x_train = train_df.iloc[: , 1:]
y_train = train_df["class"]
num_classes = len(train_df["class"].unique())
print(y_train.shape)
print(type(y_train))
x_test = test_df.iloc[: , 1:]
y_test = test_df["class"]

#ROCKET      
if rocket == True:
    print("FROCKET: ROCKET function applied on all the data, BEFORE SHARDING")
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    x_train_rocket = from_2d_array_to_nested(x_train)
    x_test_rocket = from_2d_array_to_nested(x_test)

    num_kernels = kernels

    rocket = Rocket(num_kernels, random_state=seed)

    rocket.fit(x_train_rocket)
    x_train_transform = rocket.transform(x_train_rocket)
    x_test_transform = rocket.transform(x_test_rocket)
    scaler = StandardScaler()
    X_train_transform = scaler.fit_transform(x_train_transform)
    X_test_transform = scaler.transform(x_test_transform)

    x_train = X_train_transform
    x_test = X_test_transform
    print("ROCKET SHAPES", x_train.shape, x_test.shape)

# Crea un dizionario con i dati necessari da includere nel file YAML
yaml_data = {
    'settings': {
        'listen_host': '0.0.0.0',
        'listen_port': 50051,
        'sample_shape': list(x_train.shape),
        'target_shape': list(y_train.shape) + [num_classes]
    }
}


print("Modifying director_config.yaml shapes")
with open('director_config.yaml', 'r') as yaml_file:
    yaml_data = yaml.load(yaml_file, Loader=PrettySafeLoader)

# Modifica il file YAML con le dimensioni ottenute
yaml_data['settings']['sample_shape'] = list(x_train.shape)
yaml_data['settings']['target_shape'] = list(y_train.shape) + [num_classes]

# Sovrascrivi il file YAML con i dati modificati
with open('director_config.yaml', 'w') as yaml_file:
    yaml.dump(yaml_data, yaml_file, default_flow_style=False)

print("Starting director...")
os.system("sh start_director.sh")