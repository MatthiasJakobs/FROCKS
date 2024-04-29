import os
import torch
import torchvision
import torch.nn.functional as F
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import time
import pickle
import tarfile

from torchvision import utils
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LogisticRegressionCV, RidgeClassifierCV
from sklearn.metrics import f1_score
from sktime.transformations.panel.rocket import Rocket
from sklearn.utils.validation import check_array
from sktime.datatypes._panel._convert import from_2d_array_to_nested
from sklearn.decomposition import PCA
from warnings import simplefilter
simplefilter(action='ignore')

parser = argparse.ArgumentParser(description='Reproducing experiments of the ROCKET paper')
parser.add_argument('--use_wandb', action='store_true', help='Specify whether to use Weights and Biases (wandb)')
args = parser.parse_args()

if args.use_wandb:
    import wandb

seed = [0, 1, 2, 3, 4]
data_path = "UCRArchive_2018"
datasets = os.listdir(f"{data_path}")
datasets.remove(".ipynb_checkpoints")
ordered_datasets = sorted(datasets)
datasets = ordered_datasets

if args.use_wandb:
    wandb.init()
    # Elenca i nomi degli esperimenti eseguiti
    run_names = []
    # Recupera tutti gli esperimenti nel progetto specificato
    runs = wandb.Api().runs("mlgroup/PROVA UCRArchive TCS")
    for run in runs:
        run_names.append(run.name)
    print("RUN NAMES DONE: ", run_names)
    wandb.finish()
    
for i in datasets:
    print("DATASET: ", i)
    for myseed in seed:
        print("SEED: ", myseed)
        torch.manual_seed(myseed)
        np.random.seed(myseed)
        generator=torch.Generator()
        generator.manual_seed(myseed)
        
        entity = "mlgroup"
        project = "PROVA UCRArchive TCS"
        run_name = f"{i}_{myseed}"
        tags = ["TCS"]

        if args.use_wandb:
            if run_name in run_names:
                print(f"L'esperimento {run_name} è già stato eseguito.")
                #wandb.finish()
            else:
                wandb.init(project=project, entity=entity, name=run_name, tags=tags)
        train_df = pd.read_table("UCRArchive_2018/" + i + "/" + i + "_TRAIN.tsv", header=None)
        train_df = train_df.rename(columns={0: 'class'})
        test_df = pd.read_table("UCRArchive_2018/" + i + "/" + i + "_TEST.tsv", header=None)
        test_df = test_df.rename(columns={0: 'class'})

        if train_df.isna().any().any():
            train_df = train_df.dropna(axis = 1, how='any') #drop NaN by columns
            train_df = train_df.dropna(axis = 0, how='any') #drop NaN by rows

        test_df = test_df[train_df.columns.intersection(test_df.columns)] #now test set will have the same columns of train set after removing columns
        if test_df.isna().any().any():
            test_df = test_df.dropna(axis = 0, how='any') #remove missing values by rows, because columns must be the same of train set

        # Normalization
        #scaler = StandardScaler()
        #scaler = RobustScaler()
        #train_df.iloc[: , 1:] = scaler.fit_transform(train_df.iloc[: , 1:])
        #test_df.iloc[: , 1:] = scaler.transform(test_df.iloc[: , 1:])

        if len(train_df["class"].unique()) == 2:
            train_df[train_df["class"] == -1] = 0
        if len(test_df["class"].unique()) == 2:
            test_df[test_df["class"] == -1] = 0

        X_train = train_df.iloc[: , 1:]
        y_train = train_df["class"]
        X_test = test_df.iloc[: , 1:]
        y_test = test_df["class"]

        X_train = torch.tensor(X_train.values.astype(np.float32))
        if len(train_df["class"].unique()) > 2:
            y_train = torch.tensor((y_train.values-train_df["class"].min()).astype(np.float32)) #-1 because labels start from 1
        else:
            y_train = torch.tensor((y_train.values).astype(np.float32))
        train_tensor = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_tensor, batch_size = 8, num_workers=2, drop_last=True, shuffle=False, generator=generator)

        X_test = torch.tensor(X_test.values.astype(np.float32))
        if len(test_df["class"].unique()) > 2:
            y_test = torch.tensor((y_test.values-test_df["class"].min()).astype(np.float32)) #-1 because labels start from 1
        else:
            y_test = torch.tensor((y_test.values).astype(np.float32))
        test_tensor = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_tensor, batch_size = 8, num_workers=2, drop_last=True, shuffle=False, generator=generator)

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(dev)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        in_channels = X_train.shape[1]
        num_classes = len((y_train).unique())

        class MLP(nn.Module):
            def __init__(self, in_channels, num_classes):
                super(MLP, self).__init__()
                torch.manual_seed(myseed)
                self.relu = nn.ReLU()
                self.dropout1 = nn.Dropout(0.1)
                self.dropout2 = nn.Dropout(0.2)
                self.dropout3 = nn.Dropout(0.3)
                self.ln1 = nn.Linear(in_channels, 500)
                self.ln2 = nn.Linear(500, 500)
                self.ln3 = nn.Linear(500, 500)
                self.ln4 = nn.Linear(500, num_classes)

            def forward(self, x):
                #out = self.dropout1(x)
                out = self.ln1(x)
                out = self.relu(out)
                out = self.dropout1(out)
                out = self.ln2(out)
                out = self.relu(out)
                out = self.dropout2(out)
                out = self.ln3(out)
                out = self.relu(out)
                out = self.dropout3(out)
                out = self.ln4(out)

                return out

        if num_classes == 2:
            model = MLP(in_channels, num_classes-1)
        else:
            model = MLP(in_channels, num_classes)
        print(model)

        def init(layer):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight.data)
                nn.init.constant_(layer.bias.data, 0)


        # Define an optimizer
        import torch.optim as optim
        optimizer = optim.Adam(model.parameters(), lr = 0.001)
        # Define a loss
        if num_classes == 2:
            criterion = nn.BCEWithLogitsLoss() 
        else:
            criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5, min_lr = 1e-8)

        model.apply(init)


        def train(net, loaders, optimizer, criterion, epochs=100, rocket = False, kernels = 100, dev=dev, save_param = True, model_name=i):
            torch.manual_seed(myseed)
            try:
                net = net.to(dev)
                # Initialize history
                history_loss = {"train": [], "test": []}
                history_accuracy = {"train": [], "test": []}
                history_f1 = {"train": [], "test": []}

                # Process each epoch
                for epoch in range(epochs):
                    start_epoch_time = time.time()
                    # Initialize epoch variables
                    sum_loss = {"train": 0, "test": 0}
                    sum_accuracy = {"train": 0, "test": 0}
                    sum_f1 = {"train": 0, "test": 0}

                    # Process each split
                    for split in ["train", "test"]:
                        if split == "train":
                          net.train()
                        else:
                          net.eval()
                        # Process each batch
                        for (features, labels) in loaders[split]:
                            # Move to CUDA
                            features = features.to(dev)
                            labels = labels.to(dev)
                            # Reset gradients
                            optimizer.zero_grad()
                            # Compute output
                            pred = net(features)
                            if num_classes == 2:
                                pred = pred.double()
                            labels = labels.type(torch.LongTensor)
                            labels = labels.to(dev)
                            if num_classes == 2:
                                labels = labels.unsqueeze(1)
                                labels = labels.float()
                            loss = criterion(pred, labels)
                            # Update loss
                            sum_loss[split] += loss.item()
                            # Check parameter update
                            if split == "train":
                                # Compute gradients
                                loss.backward()
                                # Optimize
                                optimizer.step()
                            # Compute accuracy
                            if num_classes == 2:
                                pred_labels = (pred >= 0).long() # Binarize predictions to 0 and 1
                                batch_accuracy = (pred_labels == labels).sum().item()/features.size(0)
                            else:
                                _,pred_labels = pred.max(1)
                                batch_accuracy = (pred_labels == labels).sum().item()/features.size(0)
                            # Update accuracy
                            sum_accuracy[split] += batch_accuracy
                            if epoch != 0:
                                if num_classes == 2:
                                    sum_f1[split] += f1_score(labels.cpu().numpy(), pred_labels.cpu().numpy(), average='binary', zero_division=0)
                                else:
                                    sum_f1[split] += f1_score(labels.cpu().numpy(), pred_labels.cpu().numpy(), average='macro', zero_division=0)
                                
                            # Store all real_labels and predictions for afterwards eval
                        scheduler.step(sum_loss["test"])
                    # Compute epoch loss/accuracy
                    epoch_loss = {split: sum_loss[split]/len(loaders[split]) for split in ["train", "test"]}
                    epoch_accuracy = {split: sum_accuracy[split]/len(loaders[split]) for split in ["train", "test"]}
                    if epoch != 0:
                        epoch_f1 = {split: sum_f1[split]/len(loaders[split]) for split in ["train", "test"]}
                    elif epoch ==0 :
                        epoch_f1 = {split: 0 for split in ["train", "test"]}
                    if args.use_wandb:
                        wandb.run.summary["step"] = epoch
                        if rocket == False:
                            wandb.log({'Training loss': epoch_loss["train"], 'Training accuracy': epoch_accuracy["train"],
                                       'Test loss': epoch_loss["test"], 'Test accuracy': epoch_accuracy["test"],
                                      'Training F1': epoch_f1["train"], 'Test F1': epoch_f1["test"]})
                        else:
                            wandb.log({f'Training loss ROCKET with {kernels} kernels': epoch_loss["train"], f'Training accuracy ROCKET with {kernels} kernels': epoch_accuracy["train"],
                                       f'Test loss ROCKET with {kernels} kernels': epoch_loss["test"], f'Test accuracy ROCKET with {kernels} kernels': epoch_accuracy["test"],
                                       f'Training F1 ROCKET with {kernels} kernels': epoch_f1["train"], f'Test F1 ROCKET with {kernels} kernels': epoch_f1["test"]})


                    # Update history
                    for split in ["train", "test"]:
                        history_loss[split].append(epoch_loss[split])
                        history_accuracy[split].append(epoch_accuracy[split])
                        if epoch != 0:
                            history_f1[split].append(epoch_f1[split])


                    # Print info
                    print(f"Epoch {epoch+1}:",
                          f"TrL={epoch_loss['train']:.4f},",
                          f"TrA={epoch_accuracy['train']:.4f},",
                          f"TeL={epoch_loss['test']:.4f},",
                          f"TeA={epoch_accuracy['test']:.4f},",
                          f"Trf1={epoch_f1['train']:.4f},",
                          f"Tef1={epoch_f1['test']:.4f},",
                          f"LR={optimizer.param_groups[0]['lr']:.5f},")
            except KeyboardInterrupt:
                print("Interrupted")
            finally:        
                # Plot loss
                plt.title("Loss")
                for split in ["train", "test"]:
                    plt.plot(history_loss[split], label=split)
                plt.legend()
                plt.show()
                # Plot accuracy
                plt.title("Accuracy")
                for split in ["train", "test"]:
                    plt.plot(history_accuracy[split], label=split)
                plt.legend()
                plt.show()

        # Define dictionary of loaders
        loaders = {"train": train_loader,
                   "test": test_loader}

        # Train model
        train(model, loaders, optimizer, criterion, epochs=100, dev=dev)

        #SCIKIT MODELS
        unique_classes, class_counts = np.unique(y_train, return_counts=True)                
        if any(class_counts < 5):
            # Se la condizione è vera, imposta n_splits su Y
            logistic_classifier = LogisticRegression()
            ridge_classifier = RidgeClassifier()
        else:
            logistic_classifier = LogisticRegressionCV()
            ridge_classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
            
        logistic_classifier.fit(X_train, y_train)
        logistic_score = logistic_classifier.score(X_test, y_test)
        logistic_preds_f1 = logistic_classifier.predict(X_test)
        if num_classes == 2:
            logistic_f1score = f1_score(y_test, logistic_preds_f1, average='binary', zero_division=0)
        else:
            logistic_f1score = f1_score(y_test, logistic_preds_f1, average='macro', zero_division=0)


        ridge_classifier.fit(X_train, y_train)
        ridge_score = ridge_classifier.score(X_test, y_test)
        ridge_preds_f1 = ridge_classifier.predict(X_test)
        if num_classes == 2:
            ridge_f1score = f1_score(y_test, ridge_preds_f1, average='binary', zero_division=0)
        else:
            ridge_f1score = f1_score(y_test, ridge_preds_f1, average='macro', zero_division=0)

        if args.use_wandb:
            wandb.log({"logistic_score": logistic_score, "ridge_score": ridge_score, 
                       "F1 logistic_score": logistic_f1score, "F1 ridge_score": ridge_f1score})

        X_train_rocket = from_2d_array_to_nested(X_train)
        X_test_rocket = from_2d_array_to_nested(X_test)

        num_kernels = [100, 1000, 10000]
        for k in num_kernels:
            rocket = Rocket(num_kernels = k, random_state=myseed)

            rocket.fit(X_train_rocket)
            X_train_transform = rocket.transform(X_train_rocket)
            X_test_transform = rocket.transform(X_test_rocket)

            if any(class_counts < 5):
                # Se la condizione è vera, imposta n_splits su Y
                rocket_logistic = LogisticRegression()
                rocket_ridge = RidgeClassifier()
            else:
                rocket_logistic = LogisticRegressionCV()
                rocket_ridge = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
            
            rocket_logistic.fit(X_train_transform, y_train)
            rocket_logistic_score = rocket_logistic.score(X_test_transform, y_test)
            rocket_logistic_preds_f1 = rocket_logistic.predict(X_test_transform)
            if num_classes == 2:
                rocket_logistic_f1score = f1_score(y_test, rocket_logistic_preds_f1, average='binary', zero_division=0)
            else:
                rocket_logistic_f1score = f1_score(y_test, rocket_logistic_preds_f1, average='macro', zero_division=0)

            rocket_ridge.fit(X_train_transform, y_train)
            rocket_ridge_score = rocket_ridge.score(X_test_transform, y_test)
            rocket_ridge_preds_f1 = rocket_ridge.predict(X_test_transform)
            if num_classes == 2:
                rocket_ridge_f1score = f1_score(y_test, rocket_ridge_preds_f1, average='binary', zero_division=0)
            else:
                rocket_ridge_f1score = f1_score(y_test, rocket_ridge_preds_f1, average='macro', zero_division=0)

            if args.use_wandb:
                wandb.log({f"logistic_rocket_score_{k}_kernels": rocket_logistic_score, 
                           f"ridge_rocket_score_{k}_kernels": rocket_ridge_score,
                           f"logistic_rocket_f1score_{k}_kernels": rocket_logistic_f1score,
                           f"ridge_rocket_f1score_{k}_kernels": rocket_ridge_f1score})

            #ROCKET extracts too many features: model does not learn. I use PCA with the explained variance ratio cumulative to extract the most important features
            pca = PCA()
            pca.fit(X_train_transform)
            explained_variance_ratio_cumulative = np.cumsum(pca.explained_variance_ratio_)
            target_variance = 0.95
            n_components = np.argmax(explained_variance_ratio_cumulative >= target_variance) + 1
            pca = PCA(n_components=n_components)
            pca.fit(X_train_transform)
            # Applica la riduzione della dimensionalità sia al set di addestramento che a quello di test
            X_train_rocket_pca = pca.transform(X_train_transform)
            X_test_rocket_pca = pca.transform(X_test_transform)

            X_train_rocket_pca = torch.tensor(X_train_rocket_pca.astype(np.float32)) 
            train_tensor = TensorDataset(X_train_rocket_pca, y_train) 
            train_loader = DataLoader(train_tensor, batch_size = 8, num_workers=2, drop_last=True, shuffle=False, generator=generator)

            X_test_rocket_pca = torch.tensor(X_test_rocket_pca.astype(np.float32)) 
            test_tensor = TensorDataset(X_test_rocket_pca, y_test) 
            test_loader = DataLoader(test_tensor, batch_size = 8, num_workers=2, drop_last=True, shuffle=False, generator=generator)

            in_channels = n_components

            if num_classes == 2:
                model = MLP(in_channels, num_classes-1)
            else:
                model = MLP(in_channels, num_classes)
            print(model)

            # Define an optimizer
            import torch.optim as optim
            optimizer = optim.Adam(model.parameters(), lr = 0.001)
            # Define a loss
            if num_classes == 2:
                criterion = nn.BCEWithLogitsLoss() 
            else:
                criterion = nn.CrossEntropyLoss()
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5, min_lr = 1e-8)

            model.apply(init)

            # Define dictionary of loaders
            loaders = {"train": train_loader,
                       "test": test_loader}

            # Train model
            train(model, loaders, optimizer, criterion, epochs=100, rocket = True, kernels = k, dev=dev)           

        if args.use_wandb:
            wandb.finish()
